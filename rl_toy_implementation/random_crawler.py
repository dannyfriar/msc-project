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


##-----------------------------------------------------------
##-------- RL Functions -------------------------------------
def get_reward(url, company_urls):
	"""Return 1 if company URL, 0 otherwise"""
	if any(c in url for c in company_urls):
		reward = 1
	else:
		reward = 0
	return reward


##-----------------------------------------------------------
##-----------------------------------------------------------
def main():
	##-------------------- Read in data
	# Company i.e. reward URLs
	companies_df = pd.read_csv('../data/domains_clean.csv')
	companies_df = companies_df[companies_df['vert_code'] <= 69203]
	companies_df = companies_df[companies_df['vert_code'] >= 69101]
	reward_urls = companies_df['url'].tolist()
	reward_urls = [l.replace("http://", "").replace("https://", "") for l in reward_urls]

	# Rest of URLs to form the state space
	first_hop_df = pd.read_csv('data/first_hop_links.csv', names = ["url"])
	url_list = reward_urls + first_hop_df['url'].tolist()
	second_hop_df = pd.read_csv('data/second_hop_links.csv', names = ["url"])
	url_list = url_list + second_hop_df['url'].tolist()
	third_hop_df = pd.read_csv('data/third_hop_links.csv', names = ["url"])
	url_list = url_list + third_hop_df['url'].tolist()
	url_list = list(set(url_list))
	del companies_df, first_hop_df, second_hop_df, third_hop_df

	# Remove any pages that obviously won't have hyperlinks/rewards
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
			r = get_reward(url, reward_urls)
			pages_crawled += 1
			total_reward += r

			# Move to next URL
			link_list = get_list_of_links(url)
			link_list = set(link_list).intersection(url_set)
			link_list = list(link_list - set(recent_urls))
			if len(link_list) == 0:
				terminal_states += 1
				break
			url = random.choice(link_list)


	print("\nCrawled {} pages, total reward = {}, # terminal states = {}"\
		.format(pages_crawled, total_reward, terminal_states))

	##----------------- Save results
	results_df = pd.DataFrame.from_dict(results_dict)
	results_df.to_csv("results/random_crawler_results_new.csv", header=True, index=False)



if __name__ == "__main__":
	main()