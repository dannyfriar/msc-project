import os
import sys
import re
import csv
import time
import random
import ahocorasick
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import OrderedDict

from evolutionai import StorageEngine
storage = StorageEngine("/nvme/webcache/")

##-----------------------------------------------------------
##-------- Miscellaneous Functions --------------------------
def progress_bar(value, endvalue, bar_length=20):
    """Print progress bar to the console"""
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

def load_csv_to_list(file_name):
	"""Loads single column CSV as list"""
	with open(file_name) as f:
		reader = csv.reader(f)
		return list(reader)[0]

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


##-----------------------------------------------------------
##-------- Get links from LMDB data functions ---------------
def get_list_of_links(url, s=storage):
	"""Use the LMDB database to get a list of links for a given URL"""
	try:
		page = s.get_page(url)
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
		# progress_bar(idx+1, len(url_list))
		try:
			link_list = get_list_of_links(url)
		except (UnicodeError, IndexError):
			skipped_urls.append(url)
		full_link_list = full_link_list + link_list
	full_link_list = full_link_list + url_list
	full_link_list = list(set(full_link_list))
	# print("\nSkipped %d URLs" % len(skipped_urls))
	return full_link_list

def lookup_domain_name(links_df, domain_url):
	"""Returns list of all URLs within domain web site (in database)"""
	return links_df[links_df['domain'] == domain_url]['url'].tolist()


##-----------------------------------------------------------
##-------- RL Functions -------------------------------------
def get_reward(url, A_company, company_urls):
	"""Return 1 if company URL, 0 otherwise"""
	if sum(check_strings(A_company, company_urls, url)) > 0:
		return 1
	return 0


##-----------------------------------------------------------
##-----------------------------------------------------------
def main():
	##-------------------- Read in data
	# Company i.e. reward URLs
	companies_df = pd.read_csv('../data/domains_clean.csv')
	companies_df = companies_df[companies_df['vert_code'] <= 69203]
	companies_df = companies_df[companies_df['vert_code'] >= 69101]
	reward_urls = companies_df['url'].tolist()
	reward_urls = [l.replace("http://", "").replace("https://", "").replace("www.", "") for l in reward_urls]
	A_company = init_automaton(reward_urls)  # Aho-corasick automaton
	A_company.make_automaton()

	# Rest of URLs to form the state space (remove any pages that obviously won't have hyperlinks/rewards)
	links_df = pd.read_csv('data/links_dataframe.csv')
	url_list = links_df['url'].tolist()
	url_list = [l.replace("http://", "").replace("https://", "") for l in url_list if type(l) is str if l[-4:] not in [".png", ".jpg", ".pdf", ".txt"]]
	url_set = set(url_list)
	
	##-------------------- Random crawling
	# Results dict for plotting
	results_dict = OrderedDict()
	results_dict['pages_crawled'] = []
	results_dict['total_reward'] = []
	results_dict['terminal_states'] = [] 

	# Parameters
	cycle_freq = 50
	number_crawls = 50000
	print_freq = 1000

	# To store
	pages_crawled = 0
	total_reward = 0
	terminal_states = 0
	count_idx = 0
	recent_urls = []
	reward_pages = []
	reward_domain_set = set()

	while count_idx < number_crawls:
		url = random.choice(list(url_set - set(recent_urls)))  # don't start at recent URL

		while count_idx < number_crawls:
			count_idx += 1

			# Track progress
			progress_bar(count_idx, number_crawls)
			if count_idx % print_freq == 0:
				print("\nCrawled {} pages, total reward = {}, # terminal states = {}"\
				.format(pages_crawled, total_reward, terminal_states))

			results_dict['pages_crawled'].append(pages_crawled)
			results_dict['total_reward'].append(total_reward)
			results_dict['terminal_states'].append(terminal_states)

			# Keep track of recent URLs (to avoid loops)
			recent_urls.append(url)
			if len(recent_urls) > cycle_freq:
				recent_urls = recent_urls[-cycle_freq:]

			# Get rewards
			r = get_reward(url, A_company, reward_urls)
			pages_crawled += 1
			total_reward += r
			if r > 0:
				reward_pages.append(url)
				reward_domain = url.split("/", 1)[0]
				reward_domain_set.update(lookup_domain_name(links_df, reward_domain))
				url_set = url_set - reward_domain_set

			# List of next possible URLs 
			link_list = get_list_of_links(url)
			link_list = set(link_list).intersection(url_set)
			link_list = list(link_list - set(recent_urls))

			# Choose next URL from list
			if r > 0 or len(link_list) == 0:
				terminal_states += 1
				break
			url = random.choice(link_list)


	print("\nCrawled {} pages, total reward = {}, # terminal states = {}"\
		.format(pages_crawled, total_reward, terminal_states))

	##----------------- Save results
	results_df = pd.DataFrame.from_dict(results_dict)
	results_df.to_csv("results/random_crawler_results_new.csv", header=True, index=False)

	df = pd.DataFrame(reward_pages, columns=["rewards_pages"])
	df.to_csv('results/random_reward_pages.csv', index=False)


if __name__ == "__main__":
	main()