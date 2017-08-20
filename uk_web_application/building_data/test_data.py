import os
import sys
import lmdb
import random
import numpy as np
import pandas as pd

from evolutionai import StorageEngine
from url_normalize import url_normalize
from urllib.parse import urlsplit

s1 = StorageEngine("/nvme/rl_project_webcache/")
s2 = StorageEngine("/nvme/uk-web/")
s3 = StorageEngine("/nvme/uk-web-backup/")

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

def extract_domain(url):
	return url.split("//")[-1].split("/")[0]

def get_uk_web_links(url, s):
	link_list = get_list_of_links(url, s)
	if len(link_list) > 0:
		return link_list
	domain_url = extract_domain(url)
	if len(domain) > 0:
		return get_list_of_links(domain, s)
	return []


#-------------------------- RL toy problem case
sample_size = 10000

# # Read in data
# links_df = pd.read_csv("../../rl_toy_implementation/new_data/links_dataframe.csv")
# rm_list = ['aarp.org', 'akc.org', 'alcon.com', 'lincoln.com', 'orlakiely.com', 
# 'red.com', 'ef.com', 'ozarksfirst.com']
# links_df['domain'] = links_df.domain.str.replace("www.", "")
# links_df = links_df[~links_df['domain'].isin(rm_list)]
# url_set = set(links_df['url'].tolist())

# def get_num_links_in_graph(url, s, url_set):
# 	link_list = get_list_of_links(url, s)
# 	link_list = list(set(link_list).intersection(url_set))
# 	return len(link_list)

# # Random sample of URLs from the database
# print("#-------- Running for RL webcache...")
# rl_url_list = []

# env1 = lmdb.open("/nvme/rl_project_webcache/", readonly=True);
# with env1.begin() as txn:
# 	cursor = txn.cursor();
# 	for key, value in cursor:
# 		rl_url_list.append(key.decode('utf-8'))

# rl_url_sample = random.sample(rl_url_list, sample_size)
# link_list_len = [get_num_links_in_graph(url, s1, url_set) for url in rl_url_sample]
# print("Mean length = {}".format(np.mean(np.array(link_list_len))))
# print("Median length = {}".format(np.median(np.array(link_list_len))))
# print("Stdev length = {}".format(np.std(np.array(link_list_len))))
# pd.DataFrame.from_dict({'num_links': link_list_len}).to_csv('test_results/rl_web_graph_links.csv')


# #-------------------------- UK web case
# def get_num_uk_links(url, s):
# 	link_list = get_list_of_links(url, s)
# 	link_list = [l for l in link_list if "uk" in l]
# 	return len(link_list)

# # Random sample of URLs from the database
# print("#-------- Running for UK web...")
# uk_url_list = []

# count = 0
# env2 = lmdb.open("/nvme/uk_web/", readonly=True);
# with env2.begin() as txn:
# 	cursor = txn.cursor();
# 	for key, value in cursor:
# 		count += 1
# 		uk_url_list.append(key.decode('utf-8'))
# 		if count >= 100000:
# 			break

# uk_url_list = random.sample(uk_url_list, sample_size)
# link_list_len = [get_num_uk_links(url, s2) for url in uk_url_list]
# print("Mean length = {}".format(np.mean(np.array(link_list_len))))
# print("Median length = {}".format(np.median(np.array(link_list_len))))
# print("Stdev length = {}".format(np.std(np.array(link_list_len))))
# pd.DataFrame.from_dict({'num_links': link_list_len}).to_csv('test_results/uk_links.csv')



#----------------- Testing extract domain function
# url_list = ['shedworking.co.uk/2011/09/', 'store.lexisnexis.co.uk/categories/accountancy', 'www.hay-kilner.co.uk/people/lucy-gray/',
# 'miceman.blogspot.co.uk/2010/12/', 'www.bbc.co.uk/news/uk-20264520']

# for url in url_list:
# 	print(url)
# 	print(extract_domain(url))


print(len(extract_domain('http:/blog.policy.manchester.ac.uk//')))


# #---------------------- Check if page text is available
# count = 0; uk_url_list = []
# env2 = lmdb.open("/nvme/uk_web", readonly=True)
# with env2.begin() as txn:
# 	cursor = txn.cursor()
# 	for key, value in cursor:
# 		uk_url_list.append(key.decode('utf-8'))
# 		count += 1
# 		if count >= 1000:
# 			break

# url = random.choice(uk_url_list)
# print(s2.get_page(url))


# print(get_list_of_links(url, s2))
# print(get_list_of_links('tinyoffice.co.uk', s3))
# print(s2.get_page('http://www.independent.co.uk/'))
# print(s3.get_page('http://www.tinyoffice.co.uk'))
# print(s2.get_page('http://www.google.co.uk'))


# ## Testing get_page function
# url = 'dating.independent.co.uk'
# env = lmdb.open('/nvme/uk-web/', readonly=True)
# with env.begin(write=False) as txn:
# 	url = url_normalize(url)
# 	print(url)
# 	payload = txn.get(url.encode('UTF-8'))
# 	print(payload)

















