import os
import sys
import lmdb
import random
import numpy as np
import pandas as pd

from evolutionai import StorageEngine
s1 = StorageEngine("/nvme/rl_project_webcache/")
s2 = StorageEngine("/nvme/uk-web/")
s3 = StorageEngine("/nvme/webcache_old/")

def get_list_of_links(url, s):
	"""Use the LMDB database to get a list of links for a given URL"""
	url = url.replace('https://', '')
	url = url.replace('http://', '')
	url = url.replace('www.', '')
	url = url.rstrip('/')
	try:
		page = s.get_page(url)
		if page is None:
			page = s.get_page(url+"/")
		if page is None:
			page = s.get_page("www."+url)
		if page is None:
			page = s.get_page("www."+url+"/")
		if page is None:
			page = s.get_page("http://"+url)
		if page is None:
			page = s.get_page("http://"+url+"/")
		if page is None:
			page = s.get_page("https://"+url)
		if page is None:
			page = s.get_page("https://"+url+"/")
		if page is None:
			page = s.get_page("http://www."+url)
		if page is None:
			page = s.get_page("http://www."+url+"/")
		if page is None:
			page = s.get_page("https://www."+url)
		if page is None:
			page = s.get_page("https://www."+url+"/")
		if page is None:
			return []
	except (UnicodeError, ValueError):
		print("Exception")
		return []
	try:
		link_list = [l.url.replace("http://", "").replace("https://", "") for l in page.links if l.url[:4] == "http"]
		link_list = link_list + [l.replace("www.", "") for l in link_list]
	except UnicodeDecodeError:
		return []
	return link_list

# #-------------------------- To send to crawler
# sample_size = 1000

# # Random sample of URLs from the database
# print("#-------- Running for UK web...")
# uk_url_list = []

# count = 0
# env2 = lmdb.open("/nvme/uk-web/", readonly=True);
# with env2.begin() as txn:
# 	cursor = txn.cursor();
# 	for key, value in cursor:
# 		count += 1
# 		uk_url_list.append(key.decode('utf-8'))
# 		if count >= sample_size:
# 			break
# pd.DataFrame.from_dict({'urls': uk_url_list}).to_csv('uk_links.csv', index=False, header=False)


#------------------------------ Testing
reward_urls = pd.read_csv("../../rl_toy_implementation/new_data/company_urls.csv")
reward_urls = [l.replace("www.", "") for l in reward_urls['url'].tolist()]
reward_urls = [l for l in reward_urls if ".uk" in l]

while True:
	url = random.choice(reward_urls)
	print(s3.get_page(url))
	input("Press enter to continue....")









