#!/usr/bin/env python3
import asyncio
import gzip
import collections
import os
import sys
import ujson as json
import zstd
import capnp
import argparse
import random
import uuid
import signal
import string
import lmdb
import glob
import time

from evolutionai import webpage_capnp
from evolutionai import StorageEngine

from nats_stream.aio.client import StreamClient
from nats_stream.aio.publisher import Publisher
from nats_stream.aio.client import StreamClient, Msg
from nats_stream.aio.subscriber import Subscriber

dctx = zstd.ZstdDecompressor()

whitelist = ["net", "org", "com", "uk"]
tld_counter = collections.Counter()

def progress_bar(value, endvalue, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

def stream_file(path, topic, client_id):
	global env, dctx
	cctx = zstd.ZstdCompressor(level=22,write_content_size=True)
	f = gzip.open(path, "rt")
	count = 0
	for line in f:
		count += 1
		progress_bar(count, 800000)
		if line[0] != "{":
			continue
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

		compressed = cctx.compress(payload)
		page = webpage_capnp.Page.from_bytes(payload)

		with env.begin(write=True) as txn:
			txn.put(page.url[:500].encode("UTF-8"), compressed)

def main():
	client_id = str(uuid.uuid4())[:8]
	parser = argparse.ArgumentParser()
	parser.add_argument("dir")
	parser.add_argument("db_path")
	# parser.add_argument("num_files")
	args = parser.parse_args()

	global env
	env = lmdb.open(args.db_path, map_size=1024**4)
	file_list = sorted(os.listdir(args.dir))
	file_list = [l for l in file_list if "gz" in l]
	# file_list = file_list[:int(args.num_files)]
	# file_list = random.sample(file_list, int(args.num_files))

	for idx, file in enumerate(file_list):
		t0 = time.time()
		filename = args.dir+ "/" + file
		# progress_bar(idx+1, len(file_list))
		try:
			stream_file(filename, "links", client_id)
		except EOFError:
			print("\n EOF error.")
			pass
		print(time.time()-t0)
	print("\n")


if __name__ == "__main__":
	main()

