import sys
import lmdb
import string
import random
import ahocorasick
import numpy as np
import pandas as pd

from urllib.parse import urlparse
from evolutionai import StorageEngine

DB_PATH = "/nvme/webcache/"
storage = StorageEngine(DB_PATH)

url = "www.eulawanalysis.blogspot.co.uk/p/studying-eu-law-catherine-barnard.html"
page = storage.get_page(url)

env = lmdb.open(DB_PATH, map_size=1024**4)
with env.begin() as txn:
	link_list = [key.decode('utf-8') for key, _ in txn.cursor()]
link_list = list(set(link_list))

print(link_list[:10])