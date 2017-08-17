import os
import sys
import lmdb
import random
import numpy as np
import pandas as pd

from evolutionai import StorageEngine
s1 = StorageEngine("/nvme/rl_project_webcache/")
s2 = StorageEngine("/nvme/uk_web/")

def get_list_of_links(url, s):
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
		return []
	try:
		link_list = [l.url.replace("http://", "").replace("https://", "") for l in page.links if l.url[:4] == "http"]
		link_list = link_list + [l.replace("www.", "") for l in link_list]
	except UnicodeDecodeError:
		return []
	return link_list

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


url = 'treasuretrails.co.uk'
print(s2.get_page('http://www.bottonline.co.uk/'))
print(get_list_of_links('www.bottonline.co.uk', s2))





