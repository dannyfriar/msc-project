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
		link_list = [l.url for l in page.links if l.url[:4] == "http"]
		link_list = [l.replace("http://", "") for l in link_list]
		link_list = [l.replace("https://", "") for l in link_list]
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
	return 1 if url in company_urls else 0



##-----------------------------------------------------------
##-------- DQN Agent ----------------------------------------
class CrawlerAgent(object):

	def __init__(self, url_list, reward_urls, 
		train_save_location = "results/dqn_crawler_train_results.csv",
		results_save_location = "results/dqn_crawler_results.csv",
		tf_model_folder = "models",
		cycle_freq=5, num_steps=10000, print_freq=1000):

		# Set up state space
		self.url_list = url_list
		self.reward_urls = reward_urls
		self.cycle_freq = cycle_freq
		self.num_steps = num_steps
		self.print_freq = print_freq

		# Set up train results dict
		self.train_results_dict = OrderedDict()
		self.train_results_dict['pages_crawled'] = []
		self.train_results_dict['total_reward'] = []
		self.train_results_dict['terminal_states'] = [] 
		self.crawled_pages = []

		# Crawler results dict
		self.results_dict = self.train_results_dict


	def train_crawler(self):
		"""Start crawling from a starting URL"""
		step_count = 0
		pages_crawled = 0
		total_reward = 0
		terminal_states = 0

		while step_count < self.num_steps:
			start_list = [l for l in self.url_list if l not in self.crawled_pages]  # don't start at an old URL
			url = random.choice(start_list)

			while step_count < self.num_steps:
				step_count += 1

				# Track progress
				progress_bar(step_count+1, self.num_steps)
				if step_count % self.print_freq == 0:
					print("\nCrawled {} pages, total reward = {}, # terminal states = {}"\
					.format(pages_crawled, total_reward, terminal_states))

				self.train_results_dict['pages_crawled'].append(pages_crawled)
				self.train_results_dict['total_reward'].append(total_reward)
				self.train_results_dict['terminal_states'].append(terminal_states)

				# Keep track of recent URLs (to avoid loops)
				self.crawled_pages.append(url)
				recent_urls = self.crawled_pages[-self.cycle_freq:]

				# Get rewards
				r = get_reward(url, self.reward_urls)
				pages_crawled += 1
				total_reward += r

				# Move to next URL
				link_list = get_list_of_links(url)
				link_list = [l for l in link_list if l in self.url_list if l not in recent_urls]
				if len(link_list) == 0:
					terminal_states += 1
					break
				url = random.choice(link_list)

		print("\nCrawled {} pages, total reward = {}, # terminal states = {}, errors = {}"\
			.format(pages_crawled, total_reward, terminal_states, errors))

		##----------------- Save results
		train_results_df = pd.DataFrame.from_dict(self.train_results_dict)
		train_results_df.to_csv(self.train_save_location, header=True, index=False)





##-----------------------------------------------------------
##-----------------------------------------------------------
def main():
	## Parameters
	cycle_freq = 5
	number_crawls = 10000  # no. crawled pages before stopping
	print_freq = 1000


	##-------------------- Read in data
	# Company i.e. reward URLs
	companies_df = pd.read_csv('../data/domains_clean.csv')
	companies_df = companies_df[companies_df['vert_code'] <= 69203]
	companies_df = companies_df[companies_df['vert_code'] >= 69101]
	reward_urls = companies_df['url'].tolist()

	# Rest of URLs to form the state space
	first_hop_df = pd.read_csv('data/first_hop_links.csv', names = ["url"])
	url_list = reward_urls + first_hop_df['url'].tolist()
	second_hop_df = pd.read_csv('data/second_hop_links.csv', names = ["url"])
	url_list = url_list + second_hop_df['url'].tolist()
	del companies_df, first_hop_df, second_hop_df

	# Remove any pages that obviously won't have hyperlinks/rewards
	url_list = [l for l in url_list if l[-4:] not in [".png", ".jpg", ".pdf", ".txt"]]


	##------------------- Initialize Crawler Agent
	agent = CrawlerAgent(url_list, reward_urls)
	agent.train_crawler()










if __name__ == "__main__":
	main()