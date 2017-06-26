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

def lookup_domain_name(links_df, domain_url):
	"""Returns list of all URLs within domain web site (in database)"""
	return links_df[links_df['domain'] == domain_url]['url'].tolist()

def append_backlinks(url, backlinks, link_list):
	"""Get the backlink for the URL, returns a string URL"""
	backlink =  backlinks[backlinks['url'].values == url]['back_url'].tolist()
	if len(backlink) == 0:
		return link_list
	link_list.append(backlink[0])
	return link_list

##-----------------------------------------------------------
##-------- RL Functions -------------------------------------
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

	# Read in backlinks data
	backlinks = pd.read_csv('data/backlinks_clean.csv')
	
	##-------------------- Random crawling
	cycle_freq = 50
	number_crawls = 50000
	print_freq = 1000
	term_steps = 50

	# To store data
	results_dict = OrderedDict([('pages_crawled', []), ('total_reward', []), ('terminal_states', [])])
	pages_crawled = 0; total_reward = 0; terminal_states = 0; num_steps = 0
	recent_urls = []; reward_pages = []
	reward_domain_set = set()

	if os.path.isfile("results/random_all_urls.csv"):
		os.remove("results/random_all_urls.csv")

	while num_steps < number_crawls:
		url = random.choice(list(url_set - set(recent_urls)))  # don't start at recent URL
		steps_without_terminating = 0

		while num_steps < number_crawls:
			num_steps += 1

			# Track progress
			progress_bar(num_steps, number_crawls)
			if num_steps % print_freq == 0:
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
			r, reward_url_idx = get_reward(url, A_company, reward_urls)
			pages_crawled += 1
			total_reward += r
			with open("results/random_all_urls.csv", "a") as csv_file:
				writer = csv.writer(csv_file, delimiter=',')
				writer.writerow([url, r])
			if r > 0:
				reward_pages.append(url)
				# reward_domain_set.update(lookup_domain_name(links_df, reward_urls[reward_url_idx]))
				# reward_urls.pop(reward_url_idx)
				# A_company = init_automaton(reward_urls)  # Aho-corasick automaton for companies
				# A_company.make_automaton()
				break

			# List of next possible URLs 
			link_list = get_list_of_links(url)
			link_list = append_backlinks(url, backlinks, link_list)
			link_list = set(link_list).intersection(url_set)
			link_list = list(link_list - set(recent_urls))

			# Choose next URL from list
			if len(link_list) == 0:
				terminal_states += 1
				break
			steps_without_terminating += 1
			if steps_without_terminating >= term_steps:
				break
			url = random.choice(link_list)


	print("\nCrawled {} pages, total reward = {}, # terminal states = {}".format(pages_crawled, total_reward, terminal_states))

	##----------------- Save results
	results_df = pd.DataFrame.from_dict(results_dict)
	results_df.to_csv("results/random_crawler_results_revisit.csv", header=True, index=False)

	df = pd.DataFrame(reward_pages, columns=["rewards_pages"])
	df.to_csv('results/random_reward_pages.csv', index=False)


if __name__ == "__main__":
	main()