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
storage = StorageEngine("/nvme/rl_project_webcache/")

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
		link_list = link_list + [l.replace("www.", "") for l in link_list]
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
links_df = pd.read_csv("new_data/links_dataframe.csv")
rm_list = ['aarp.org', 'akc.org', 'alcon.com', 'lincoln.com', 'orlakiely.com', 
'red.com', 'ef.com', 'ozarksfirst.com']
links_df['domain'] = links_df.domain.str.replace("www.", "")
links_df = links_df[~links_df['domain'].isin(rm_list)]
reward_urls = links_df[links_df['type']=='company-url']['url']
reward_urls = [l.replace("www.", "") for l in reward_urls]
A_company = init_automaton(reward_urls)  # Aho-corasick automaton
A_company.make_automaton()
url_set = set(links_df['url'].tolist())
url_list = list(url_set)

# Read random sample of URLs
# test_urls = pd.read_csv("data/random_url_sample.csv")['url'].tolist()  # random URLs
# test_urls = pd.read_csv("results/linear_dqn_results/visited_value.csv")['url'].tolist()
test_urls = pd.read_csv("results/embedding_buffer_results/predicted_value.csv")['url'].tolist()
print(len(test_urls))

gamma = 0.9
results_dict = OrderedDict([('url', []), ('true_value', [])])

for idx, url in enumerate(test_urls):
	progress_bar(idx, len(test_urls))
	results_dict['url'].append(url)

	# print("Checking reward...")
	r = sum(check_strings(A_company, reward_urls, url))
	if r >= 1:
		# print("Return of {} is {}".format(url, 1))
		results_dict['true_value'].append(1)
		# input("Press enter to continue...")
		continue

	# print("First links...")
	first_hop_links = get_list_of_links(url)
	first_hop_links = list(set(first_hop_links).intersection(url_set))
	if len(first_hop_links) == 0:
		# print("No first hop links")
		# print("Return of {} is {}".format(url, 0))
		results_dict['true_value'].append(0)
		# input("Press enter to continue...")
		continue
	if len(first_hop_links) > 100:
		results_dict['url'] = results_dict['url'][:-1]
		continue

	# print("Checking reward...")
	r = sum(check_strings(A_company, reward_urls, " ".join(first_hop_links)))
	if r >= 1:
		# print("Return of {} is {}".format(url, gamma))
		results_dict['true_value'].append(gamma)
		# input("Press enter to continue...")
		continue

	# print("Second links...")
	second_hop_links = get_all_links(first_hop_links)
	second_hop_links = list(set(second_hop_links).intersection(url_set))

	if len(second_hop_links) == 0:
		# print("No second hop links")
		# print("Return of {} is {}".format(url, 0))
		results_dict['true_value'].append(0)
		# input("Press enter to continue...")
		continue

	# print("Checking reward...")
	r = sum(check_strings(A_company, reward_urls, " ".join(second_hop_links)))
	if r >= 1:
		# print("Return of {} is {}".format(url, gamma**2))
		results_dict['true_value'].append(gamma**2)
		# input("Press enter to continue...")
		continue
	else:
		results_dict['true_value'].append(0)

	# input("Press enter to continue...")


print("\nSaving Results...")
print(len(results_dict['url']))
print(len(results_dict['true_value']))
# pd.DataFrame.from_dict(results_dict).to_csv("results/actual_value_revisit.csv", index=False)
pd.DataFrame.from_dict(results_dict).to_csv("results/embedding_buffer_results/actual_value_visited.csv", index=False)
print("Done.")