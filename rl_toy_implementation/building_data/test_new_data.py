import os
import sys
import re
import csv
import time
import random
import argparse
import ahocorasick
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import OrderedDict

from evolutionai import StorageEngine
storage = StorageEngine("/nvme/rl_project_webcache/")

#-----------------------------------------------------------
def init_automaton(string_list):
	"""Make Aho-Corasick automaton from a list of strings"""
	A = ahocorasick.Automaton()
	for idx, s in enumerate(string_list):
		A.add_word(s, (idx, s))
	return A

def check_strings(A, search_list, string_to_search):
	"""Aho Corasick algorithm, return boolean list of strings within longer string"""
	index_list = []
	for item in A.iter(string_to_search):
		index_list.append(item[1][0])

	output_list = np.array([0] * len(search_list))
	output_list[index_list] = 1
	return output_list.tolist()


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
	except UnicodeError:
		return []
	try:
		link_list = [l.url.replace("http://", "").replace("https://", "") for l in page.links if l.url[:4] == "http"]
		link_list = link_list + [l.replace("www.", "") for l in link_list]
	except UnicodeDecodeError:
		return []
	return link_list


def get_reward(url, A_company, company_urls):
	"""Return 1 if company URL, 0 otherwise"""
	idx_list = check_strings(A_company, company_urls, url)
	if sum(idx_list) > 0:
		return 1
	return 0




# Read in data - reward URLs and all URLs in the state space
links_df = pd.read_csv("../new_data/links_dataframe.csv")
reward_urls = links_df[links_df['type']=='company-url']['url']
reward_urls = [l.replace("www.", "") for l in reward_urls]
A_company = init_automaton(reward_urls)  # Aho-corasick automaton
A_company.make_automaton()
url_set = set(links_df['url'].tolist())
url_list = list(url_set)


# Test URLs
url = "www.jdjournal.com/tag/union-membership/"
print("Reward is {}".format(get_reward(url, A_company, reward_urls)))
link_list = get_list_of_links(url)

link_list = list(set(link_list).intersection(url_set))
print(link_list)
# next_url = random.choice(link_list)
# print(next_url)

# link_list = get_list_of_links(next_url)
# print(link_list)




















