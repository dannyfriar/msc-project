import os
import sys
import re
import csv
import time
import random
import pickle
import argparse
import ahocorasick
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import OrderedDict

from evolutionai import StorageEngine
storage = StorageEngine("/nvme/webcache/")

from nltk.corpus import stopwords, words, names
stops = stopwords.words("english")

##-----------------------------
def get_list_of_links(url, s=storage):
	"""Use the LMDB database to get a list of links for a given URL"""
	try:
		page = s.get_page(url)
		print(page.links)
		# print(page)
	except UnicodeError:
		return []
	if page is None:
		return []
	try:
		link_list = [l.url.replace("http://", "").replace("https://", "") for l in page.links if l.url[:4] == "http"]
	except UnicodeDecodeError:
		return []
	return link_list

def get_all_links(url_list):
	"""Get all links from a list of URLs"""
	full_link_list = []
	skipped_urls = []
	for idx, url in enumerate(url_list):
		progress_bar(idx+1, len(url_list))
		try:
			link_list = get_list_of_links(url)
		except (UnicodeError, IndexError):
			skipped_urls.append(url)
		full_link_list = full_link_list + link_list
	full_link_list = full_link_list + url_list
	full_link_list = list(set(full_link_list))
	print("\nSkipped %d URLs" % len(skipped_urls))
	return full_link_list


def init_automaton(string_list):
	"""Make Aho-Corasick automaton from a list of strings"""
	A = ahocorasick.Automaton()
	for idx, s in enumerate(string_list):
		A.add_word(s, (idx, s))
	return A

def check_strings(A, search_list, string_to_search):
	"""Use Aho Corasick algorithm to produce boolean list indicating
	prescence of strings within a longer string"""
	index_list = []
	for item in A.iter(string_to_search):
		index_list.append(item[1][0])

	output_list = np.array([0] * len(search_list))
	output_list[index_list] = 1
	return output_list.tolist()

def get_reward(url, A_company, company_urls):
	"""Return 1 if company URL, 0 otherwise"""
	if sum(check_strings(A_company, company_urls, url)) > 0:
		return 1
	return 0


# Company i.e. reward URLs
companies_df = pd.read_csv('../data/domains_clean.csv')
companies_df = companies_df[companies_df['vert_code'] <= 69203]
companies_df = companies_df[companies_df['vert_code'] >= 69101]
reward_urls = companies_df['url'].tolist()
reward_urls = [l.replace("http://", "").replace("https://", "") for l in reward_urls]
A_company = init_automaton(reward_urls)  # Aho-corasick automaton for companies
A_company.make_automaton()

links_df = pd.read_csv('data/links_dataframe.csv')
url_list = links_df['url'].tolist()
url_list = [l.replace("http://", "").replace("https://", "") for l in url_list if type(l) is str if l[-4:] not in [".png", ".jpg", ".pdf", ".txt"]]

# Get value of a random URL
url = random.choice(url_list)





























