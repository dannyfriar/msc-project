import os
import sys
import re
import csv
import time
import zstd
import lmdb
import random
import pickle
import argparse
import threading
import ahocorasick
import multiprocessing
import numpy as np
import pandas as pd
import tensorflow as tf

from evolutionai import StorageEngine
storage = StorageEngine("/nvme/uk_web/")

def get_list_of_links(url, s=storage):
	"""Use the LMDB database to get a list of links for a given URL"""
	try:
		page = s.get_page(url)
		if page is None:
			page = s.get_page(url+"/")
		if page is None:
			page = s.get_page("www."+url)
		if page is None:
			page = s.get_page("www."+url+"/")
		if page is None:
			return []
	except (UnicodeError, ValueError):
		return []
	try:
		link_list = [l.url.replace("http://", "").replace("https://", "") for l in page.links if l.url[:4] == "http"]
		link_list = link_list + [l.replace("www.", "") for l in link_list]
	except UnicodeDecodeError:
		return []
	return link_list

env = lmdb.open('/nvme/uk_web/', readonly=True);
url_list = []

with env.begin() as txn:
	cursor = txn.cursor();
	for key, value in cursor:
		url_list.append(key)

# url = random.choice(url_list).decode('utf-8')
url = 'https://www.napit.org.uk/electricians-in-merriott.aspx'
print(url)
print(get_list_of_links(url))




