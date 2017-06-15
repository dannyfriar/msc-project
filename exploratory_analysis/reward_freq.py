# -*- coding: utf-8 -*-
import urllib.request
import os
import sys
import math
import pdb
import re
import csv
import time
import random
import ahocorasick
import numpy as np
import pandas as pd

from socket import timeout
from collections import OrderedDict, Counter
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

# Stop words (NLTK)
stops = stopwords.words("english")
stops.append('activities')
stops.append('nec')
stops = set(stops)


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
##-------- HTML/Web Scraping/Common crawl Function ----------
def find_links(url):
	"""Obtain list of links from a URL"""
	link_list = []
	full_link_list = []
	try:
		page = urllib.request.urlopen(url)
		soup = BeautifulSoup(page, 'lxml', from_encoding=page.info().get_param('charset'))
	except:
		return []
	for link in soup.find_all('a', href=True):
		link_text = link['href']
		if link_text[:4]=="http" and url not in link_text:
			link_list.append(link_text)
	return link_list


def init_automaton(string_list):
	"""Make Aho-Corasick automaton from a list of strings"""
	A = ahocorasick.Automaton()
	for idx, s in enumerate(string_list):
		A.add_word(s, (idx, s))
	return A

def check_strings(A, search_list, pagetxt):
	"""Use Aho Corasick algorithm to produce boolean list indicating
	prescence of strings within a longer string"""
	index_list = []
	for item in A.iter(pagetxt):
		index_list.append(item[1][0])

	output_list = np.array([0] * len(search_list))
	output_list[index_list] = 1
	return output_list.tolist()


##-----------------------------------------------------------
##-------- RL functions -------------------------------------
def choose_random_action(action_list):
	"""Chooses random action from a list"""
	return random.choice(action_list)


##-----------------------------------------------------------
##-------- Agent Class --------------------------------------
class Agent(object):
	def __init__(self):
		pass

##-----------------------------------------------------------
def main():

	# Load in company names, URLs and vertical keywords
	# word_list = load_csv_to_list('../data/vert_desc_words.csv')
	# word_list = ['accounting', 'auditing']
	# name_list = load_csv_to_list('../data/company_names_ltd.csv')
	# url_list = load_csv_to_list('../data/company_urls.csv')

	# Get companies from specific industry
	companies_df = pd.read_csv('../data/domains_clean.csv')
	companies_df = companies_df[companies_df['vert_code'] <= 69203]
	companies_df = companies_df[companies_df['vert_code'] >= 69101]
	companies_df = companies_df[companies_df['is_small_name'] == 0]
	name_list = companies_df['company_name_short'].tolist()
	url_list = companies_df['url'].tolist()

	vert_list = companies_df['vert_desc'].tolist()
	vert_list = [re.sub('[^a-zA-Z -]+', '', s).strip() for s in vert_list if s != "None Supplied"]
	word_list = list(" ".join(vert_list).split())
	word_list = [word.lower() for word in word_list if word not in stops and len(word) > 1]
	word_list = list(set(word_list))

	# Check frequency of words
	# word_freqs = Counter(word_list)
	# word_freqs = pd.DataFrame.from_dict(word_freqs, orient='index')
	# word_freqs.columns = ['count']
	# word_freqs = word_freqs[word_freqs['count'] >= 100]
	# word_freqs = word_freqs.sort(['count'], ascending = [0])

	# Make Aho-Corasick automatons	
	A_vert = init_automaton(word_list)
	A_vert.make_automaton()
	A_comp = init_automaton(name_list)
	A_comp.make_automaton()

	##-------------- Crawling ---------------
	pages_crawled = 0
	name_rewards = 0
	vert_rewards = 0
	crawled_pages = []
	reward_check = np.array([0] * len(name_list))
	vert_check = np.array([0] * len(word_list))

	# while True:
	for _ in range(20):
		# Choose start URL
		url = random.choice(url_list)

		while True:
			print("------ Crawled %d pages, receiving %d vert rewards and %d name rewards" 
				% (pages_crawled, vert_rewards, name_rewards))

			# Follow random link from the page
			link_list = find_links(url)
			if len(link_list) == 0:
				print("Choosing again...")
				break
			url = random.choice(link_list)

			# Check for reward within page
			try:
				pagetxt = urllib.request.urlopen(url, timeout=8).read().decode('utf-8').lower()
				bin_output = check_strings(A_comp, name_list, pagetxt)
				reward_check = reward_check + np.array(bin_output)
				bin_output_word = check_strings(A_vert, word_list, pagetxt)
				vert_check = vert_check + np.array(bin_output_word)

				if sum(bin_output) > 0:
					name_rewards += 1
				if sum(bin_output_word) > 0:
					vert_rewards += 1

				pages_crawled += 1
				crawled_pages.append(url)
				page_reward = sum(bin_output_word) / len(word_list)
				# input("Press Enter to continue...")

			except (UnicodeDecodeError, urllib.error.HTTPError, urllib.error.URLError, timeout):
				print("Excepted")
				pass

		##
		check_dict = OrderedDict()
		check_dict['name'] = name_list
		check_dict['count'] = reward_check.tolist()
		check_df = pd.DataFrame.from_dict(check_dict)
		check_df.to_csv('results/check_rewards.csv', index=False)

		vert_dict = OrderedDict()
		vert_dict['name'] = word_list
		vert_dict['count'] = vert_check.tolist()
		vert_df = pd.DataFrame.from_dict(vert_dict)
		vert_df.to_csv('results/vert_freq.csv', index=False)

		crawled_pages_dict = OrderedDict()
		crawled_pages_dict['url'] = crawled_pages
		crawled_pages_dict['url_reward'] = page_reward
		crawl_df = pd.DataFrame.from_dict(crawled_pages_dict)
		crawl_df.to_csv('results/crawled_pages.csv', index=False)


if __name__ == "__main__":
	random.seed(0)
	main()