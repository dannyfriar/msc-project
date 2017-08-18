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
import logging
import cython
import numpy as np
import pandas as pd

from evolutionai import webpage_capnp
from evolutionai import StorageEngine

from nats_stream.aio.client import StreamClient
from nats_stream.aio.publisher import Publisher
from nats_stream.aio.client import StreamClient, Msg
from nats_stream.aio.subscriber import Subscriber

logging.basicConfig(filename='log_files/load_uk_web.log', level=logging.DEBUG)
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
	cctx = zstd.ZstdCompressor(level=5, write_content_size=True)
	f = gzip.open(path, "rt")
	data_list = []; url_list = []
	error_count = 0; line_count = 0

	for count, line in enumerate(f):
		line_count += 1
		# progress_bar(count, 3000)
		if line[0] != "{":
			error_count += 1
			continue
		j = json.loads(line)
		if "WARC-Target-URI" not in j["Envelope"]["WARC-Header-Metadata"]:
			error_count += 1
			continue
		page = webpage_capnp.Page.new_message()
		page.url = j["Envelope"]["WARC-Header-Metadata"]["WARC-Target-URI"]
		tld_counter[page.url.split("/")[2].split(".")[-1]] += 1
		if not page.url.split("/")[2].endswith(tuple(whitelist)):
			continue
		try:
			page.title = j['Envelope']['Payload-Metadata']["HTTP-Response-Metadata"]['HTML-Metadata'] \
				.get('Head', {}).get('Title', "")
		except KeyError:
			page.title = 'Fake title'
		try:
			links = j['Envelope']['Payload-Metadata']["HTTP-Response-Metadata"]['HTML-Metadata'].get('Links', [])
		except KeyError:
			# print("No links...")
			links = []
		clinks = page.init("links", len(links))

		for i, d in enumerate(links):
			l = clinks[i]
			l.url = links[i].get("url", "")
			l.text = links[i].get("text", "")
			l.title = links[i].get("title", "")

		payload = page.to_bytes()
		compressed = cctx.compress(payload)
		data_list.append((page.url[:500].encode("UTF-8"), compressed))
		url_list.append(page.url[:500].encode("UTF-8"))

	with env.begin(write=True) as txn:
		save_tuple = txn.cursor().putmulti(data_list, overwrite=True)
		print(save_tuple[1])

	pd.DataFrame.from_dict({'url': url_list}).to_csv("first_file_urls.csv")


def main():
	client_id = str(uuid.uuid4())[:8]
	parser = argparse.ArgumentParser()
	parser.add_argument("dir")
	parser.add_argument("db_path")
	args = parser.parse_args()

	global env
	env = lmdb.open(args.db_path, map_size=1024**4)
	file_list = sorted(os.listdir(args.dir))
	file_list = [l for l in file_list if "gz" in l]
	# file_list = file_list[35000:]
	print("Running for directory {} with {} files".format(args.dir, len(file_list)))

	print_iters = 100
	t0 = time.time()
	for idx, file in enumerate(file_list):
		filename = args.dir+ "/" + file
		progress_bar(idx+1, len(file_list))
		input("Press enter to continue...")
		try:
			logging.info('Running for file {}...'.format(file))
			stream_file(filename, "links", client_id)
			logging.info('Completed file {}, time elapsed = {}.'.format(file, time.time()-t0))
		except EOFError:
			print("\n EOF error on file {}.".format(file))
			logging.info("Error for file {}.".format(file))
			pass
		if idx % print_iters == 0:
			print("\nRun for {} files, time elapsed = {}".format(idx+1, time.time()-t0))
	print("\n")


if __name__ == "__main__":
	main()