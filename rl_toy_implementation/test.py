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
def progress_bar(value, endvalue, bar_length=20):
    """Print progress bar to the console"""
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


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
	except UnicodeDecodeError:
		return []
	return link_list

def get_all_links(url_list):
	"""Get all links from a list of URLs"""
	full_link_list = []
	skipped_urls = []
	for idx, url in enumerate(url_list):
		# progress_bar(idx+1, len(url_list))
		try:
			link_list = get_list_of_links(url)
		except (UnicodeError, IndexError):
			skipped_urls.append(url)
			link_list = []
		full_link_list = full_link_list + link_list
	full_link_list = full_link_list + url_list
	full_link_list = list(set(full_link_list))
	# print("\nSkipped %d URLs" % len(skipped_urls))
	return full_link_list

def append_backlinks(url, backlinks, link_list):
	"""Get the backlink for the URL, returns a string"""
	backlink =  backlinks[backlinks['url'] == url]['back_url'].tolist()
	if len(backlink) == 0:
		return link_list
	link_list.append(backlink[0])
	return link_list


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

# Read in backlinks
backlinks = pd.read_csv('data/backlinks_clean.csv')

# url = 'gloopa.co.uk/england-commercial-register-letter/#comment-479'
# link_list = get_list_of_links(url)
# print(link_list)
# link_list = append_backlinks(url, backlinks, link_list)
# print("\n")
# print(link_list)
# # print(get_reward('www.dennisandturnbull.com/dt-sterling-boardrooms-business-leaders/', A_company, reward_urls))


# print([p.url for p in storage.get_pages_prefix("http://www.big")])

print(get_reward("www.lassaccounts.co.uk/lakai-c-63_221.html", A_company, reward_urls))




# Get value of a random URL
gamma = 0.5
loop_range = 1000
crawled_pages = 0
path_to_reward = 0


for idx, i in enumerate(range(loop_range)):
	# progress_bar(idx, loop_range)
	input("Press enter to continue...")
	url = random.choice(url_list)
	print(url)

	print("Checking reward...")
	r = sum(check_strings(A_company, reward_urls, url))
	if r >= 1:
		print("Return of {} is {}".format(url, r))
		# exit(0)
		continue

	# crawled_pages += 1

	print("First links...")
	first_hop_links = get_list_of_links(url)
	first_hop_links = append_backlinks(url, backlinks, first_hop_links)
	if len(first_hop_links) == 0:
		print("No first hop links")
		print("Return of {} is {}".format(url, 0))
		# exit(0)
		continue

	print("Checking reward...")
	r = sum(check_strings(A_company, reward_urls, " ".join(first_hop_links)))
	if r >= 1:
		print("Return of {} is {}".format(url, gamma*r))
		path_to_reward += 1
		continue

	print("Second links...")
	second_hop_links = get_all_links(first_hop_links)
	backlink_list = backlinks[backlinks['url'].isin(first_hop_links)]['back_url'].tolist()
	second_hop_links += backlink_list

	if len(second_hop_links) == 0:
		print("No second hop links")
		print("Return of {} is {}".format(url, 0))
		continue

	print("Checking reward...")
	r = sum(check_strings(A_company, reward_urls, " ".join(second_hop_links)))
	if r >= 1:
		print("Return of {} is {}".format(url, r*gamma**2))
		path_to_reward += 1
		continue

	print("Third links...")
	third_hop_links = get_all_links(second_hop_links)
	backlink_list = backlinks[backlinks['url'].isin(second_hop_links)]['back_url'].tolist()
	third_hop_links += backlink_list

	if len(third_hop_links) == 0:
		print("No third hop links")
		print("Return of {} is {}".format(url, 0))
		continue

	print("Checking reward...")
	r = sum(check_strings(A_company, reward_urls, " ".join(third_hop_links)))
	if r >= 1:
		print("Return of {} is {}".format(url, r*gamma**3))
		path_to_reward += 1
		continue

	print("Return of {} is {}".format(url, 0))
# print("\nPercent with path to reward {}".format(path_to_reward/crawled_pages))


