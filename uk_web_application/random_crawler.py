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
storage = StorageEngine("/nvme/uk_web/")

RESULTS_FOLDER = "results/random_results/"

##-----------------------------------------------------------
##-------- Miscellaneous Functions --------------------------
def progress_bar(value, endvalue, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

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

##-----------------------------------------------------------
##-------- Get links from LMDB data functions ---------------
def get_list_of_links(url, s=storage):
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

##-----------------------------------------------------------
##-------- RL Functions -------------------------------------
def get_random_url(url_list, recent_urls):
	"""Get random url that is not in list of recent URLs"""
	url = random.choice(url_list)
	while url in recent_urls:
		url = random.choice(url_list)
	return url

def get_reward(url, A_company, company_urls):
	"""Return 1 if company URL, 0 otherwise"""
	idx_list = check_strings(A_company, company_urls, url)
	if sum(idx_list) > 0:
		reward_url_idx = np.nonzero(idx_list)[0][0]
		return 1, reward_url_idx
	return 0, None

##-----------------------------------------------------------
##-----------------------------------------------------------
def main():
	##-------------------- Read in data - reward URLs and all URLs in the state space
	links_df = pd.read_csv("../rl_toy_implementation/new_data/links_dataframe.csv")
	rm_list = ['aarp.org', 'akc.org', 'alcon.com', 'lincoln.com', 'orlakiely.com', 
	'red.com', 'ef.com', 'ozarksfirst.com']
	links_df['domain'] = links_df.domain.str.replace("www.", "")
	links_df = links_df[~links_df['domain'].isin(rm_list)]
	reward_urls = pd.read_csv("../rl_toy_implementation/new_data/company_urls.csv")
	reward_urls = [l.replace("www.", "") for l in reward_urls['url'].tolist()]
	reward_urls = [l for l in reward_urls if ".uk" in l]
	A_company = init_automaton(reward_urls)  # Aho-corasick automaton
	A_company.make_automaton()

	start_url_list = links_df['url'].tolist()
	# start_url_list1 = links_df[links_df['type']=='first-hop-link']['url'].tolist()
	# start_url_list2 = links_df[links_df['type']=='second-hop-link']['url'].tolist()
	# start_url_list = start_url_list1 + start_url_list2
	start_url_list = [l for l in start_url_list if ".uk" in l]
	print(len(start_url_list))

	# Set paths
	all_urls_file = RESULTS_FOLDER + "all_urls.csv"
	
	##-------------------- Random crawling
	cycle_freq = 20
	number_crawls = 200000
	print_freq = 100000
	term_steps = 50

	# input("Press enter to continue...")

	if os.path.isfile(all_urls_file):
		os.remove(all_urls_file)

	pages_crawled = 0; total_reward = 0; terminal_states = 0; num_steps = 0
	recent_urls = []; reward_pages = []
	reward_domain_set = set()

	while num_steps < number_crawls:
		url = get_random_url(start_url_list, recent_urls) 
		steps_without_terminating = 0

		while num_steps < number_crawls:
			num_steps += 1

			# Track progress
			progress_bar(num_steps, number_crawls)
			if num_steps % print_freq == 0:
				print("\nCrawled {} pages, total reward = {}, # terminal states = {}"\
					.format(pages_crawled, total_reward, terminal_states, len(reward_urls)))

			# Keep track of recent URLs (to avoid loops)
			recent_urls.append(url)
			if len(recent_urls) > cycle_freq:
				recent_urls = recent_urls[-cycle_freq:]

			# Get rewards
			r, reward_url_idx = get_reward(url, A_company, reward_urls)
			pages_crawled += 1
			total_reward += r

			# List of next possible URLs 
			link_list = get_list_of_links(url)
			link_list = [l for l in link_list if ".uk" in l]
			link_list = list(set(link_list) - set(recent_urls))
			if len(link_list) == 0:
				is_terminal = 1
			else:
				is_terminal = 0
				terminal_states += 1

			with open(all_urls_file, "a") as csv_file:
				writer = csv.writer(csv_file, delimiter=',')
				writer.writerow([url, r, is_terminal])

			if r > 0 or is_terminal == 1:
				break

			# Choose next URL from list
			if len(link_list) == 0:
				terminal_states += 1
				break
			steps_without_terminating += 1
			if steps_without_terminating >= term_steps:
				break
			url = random.choice(link_list)


if __name__ == "__main__":
	main()