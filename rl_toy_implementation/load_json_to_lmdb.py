#!/usr/bin/env python3
# this script opens .warc.wat file, extracts links and streams into nats-streaming server
import asyncio
import gzip
import collections
import os
import sys
import ujson as json
import zstd
import capnp
import argparse
import uuid
import signal
import string
import lmdb
import glob

from evolutionai import webpage_capnp
from evolutionai import StorageEngine
from nats_stream.aio.client import StreamClient
from nats_stream.aio.publisher import Publisher
from nats_stream.aio.client import StreamClient, Msg
from nats_stream.aio.subscriber import Subscriber

dctx = zstd.ZstdDecompressor()
storage = StorageEngine("/nvme/webcache1/")

whitelist = ["net", "org", "com", "uk"]
tld_counter = collections.Counter()

def progress_bar(value, endvalue, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

async def stream_file(loop, path, topic,client_id):
	f = gzip.open(path, "rt")
	count_all = 0
	count_processed = 0
	for line in f:
		if line[0] != "{":
			continue
		count_all += 1
		j = json.loads(line)
		if "WARC-Target-URI" not in j["Envelope"]["WARC-Header-Metadata"]:
			continue
		if "HTTP-Response-Metadata" not in j['Envelope']['Payload-Metadata']:
			continue
		if "HTML-Metadata" not in j['Envelope']['Payload-Metadata']["HTTP-Response-Metadata"]:
			continue
		page = webpage_capnp.Page.new_message()
		page.url = j["Envelope"]["WARC-Header-Metadata"]["WARC-Target-URI"]
		tld_counter[page.url.split("/")[2].split(".")[-1]] += 1
		if not page.url.split("/")[2].endswith(tuple(whitelist)):
			continue
		page.title = j['Envelope']['Payload-Metadata']["HTTP-Response-Metadata"]['HTML-Metadata'] \
			.get('Head', {}).get('Title', "")
		links = j['Envelope']['Payload-Metadata']["HTTP-Response-Metadata"]['HTML-Metadata'].get('Links', [])
		clinks = page.init("links", len(links))
		for i, d in enumerate(links):
			l = clinks[i]
			l.url = links[i].get("url", "")
			l.text = links[i].get("text", "")
			l.title = links[i].get("title", "")
		payload = page.to_bytes()
		count_processed += 1
		return payload

def message_handler(payload):
	global env, dctx
	cctx = zstd.ZstdCompressor(level=22,write_content_size=True)
	compressed = cctx.compress(payload)
	page = webpage_capnp.Page.from_bytes(payload)
	# print(page.url)
	with env.begin(write=True) as txn:
		txn.put(page.url[:500].encode("UTF-8"), compressed)


def main():
	client_id = str(uuid.uuid4())[:8]
	parser = argparse.ArgumentParser()
	parser.add_argument("dir")
	parser.add_argument("db_path")
	args = parser.parse_args()

	global env
	env = lmdb.open(args.db_path, map_size=1024**4)
	file_list = os.listdir(args.dir)
	for idx, file in enumerate(file_list):
		filename = args.dir+ "/" + file
		progress_bar(idx+1, len(file_list))
		loop = asyncio.get_event_loop()
		payload = loop.run_until_complete(
			stream_file(
				loop,
				filename,
				"links",
				client_id)
		)
		message_handler(payload=payload)
	loop.close()
	print("\n")
	
	# print(storage.get_page('http://blog.aarp.org/tag/hipertension/'))


if __name__ == "__main__":
	main()