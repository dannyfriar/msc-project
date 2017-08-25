import os
import re
import sys
import csv
import time
import gensim
import random
import pickle
import numpy as np
import pandas as pd
import ujson as json
import tensorflow as tf

from wordsegment import segment
from urllib.request import urlopen

from nltk.corpus import stopwords, words, names
stops = set(stopwords.words("english"))
eng_words = set(words.words())

##---------------- Misc functions
def progress_bar(value, endvalue, bar_length=20):
	percent = float(value) / endvalue
	arrow = '-' * int(round(percent * bar_length)-1) + '>'
	spaces = ' ' * (bar_length - len(arrow))
	sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
	sys.stdout.flush()

def save_results_list(filename, results_list):
	"""Save results to given file"""
	with open(filename, "a") as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		writer.writerow(results_list)

##---------------- To call Wikipedia API
class WikiCrawler(object):
	def __init__(self):
		self.url_base = ''.join(['https://en.wikipedia.org/w/api.php?action=query&titles={}',
			'&prop=links&pllimit=max&format=json'])
		self.random_url = ''.join(['https://en.wikipedia.org/w/api.php?action=query&generator',
			'=random&grnnamespace=0&prop=links&pllimit=max&format=json'])
		self.title_blacklist = ['User', 'Talk', 'Wikipedia', 'Help', 'Template', 
			'Category', 'Portal', 'File']

	def title_in_blacklist(self, title):
		"""Check if title contains words in blacklist"""
		if sum([t in title for t in self.title_blacklist]) > 0:
			return True

	def get_wiki_links(self, random, url_title):
		"""Gets wikipedia URLs given page title"""
		if random == True:
			url = self.random_url
		else:
			url = self.url_base.format(url_title)
			url = url.replace(' ', '%20')
		try:
			response = urlopen(url).read().decode('UTF-8')
		except UnicodeEncodeError as e:
			return url_title, []

		response = json.loads(response)['query']['pages']
		pages_dict = response[next(iter(response))]
		title = pages_dict['title']
		try:
			links = pages_dict['links']
			title_links_list = [d['title'] for d in links]
			title_links_list = [t for t in title_links_list if not self.title_in_blacklist(t)]
		except KeyError as e:
			title_links_list = []
		return title, title_links_list

	def choose_random_title(self, title_list):
		"""Choose random title from list not in blacklist"""
		title = 'User'
		while self.title_in_blacklist(title) == True:
			title = random.choice(title_list)
		return title

##---------------- RL functions
def get_reward(title):
	"""Give reward if philosophy page found"""
	if 'philosophy' in title.lower():
		return 1
	return 0


##-----------------------------
def main():
	num_crawls = 10000
	print_freq = 100
	copy_steps = 50
	cycle_freq = 10
	term_steps = 15

	print("#------- Loading data...")
	with open('first_hops.pkl', 'rb') as f:
		first_hop_start = pickle.load(f)
	with open('second_hops.pkl', 'rb') as f:
		second_hop_start = pickle.load(f)
	# start_pages = first_hop_start + second_hop_start
	start_pages = first_hop_start

	results_file = 'random_wiki_crawl_results.csv'
	if os.path.isfile(results_file):
		os.remove(results_file)

	#---------------- Starting crawler
	print("#------- Starting crawler...")
	steps = 0; rewards = 0; terminal_states = 0; recent_titles = []
	wiki_crawl = WikiCrawler()

	while steps < num_crawls:
		# title, title_links_list = wiki_crawl.get_wiki_links(True, None)
		title, title_links_list = wiki_crawl.get_wiki_links(False, random.choice(start_pages))
		steps_before_term = 0

		while steps < num_crawls:
			is_terminal = 0
			steps += 1
			steps_before_term += 1
			progress_bar(steps, num_crawls)

			# Keep track of recent URLs (to avoid loops)
			recent_titles.append(title)
			if len(recent_titles) > cycle_freq:
				recent_titles = recent_titles[-cycle_freq:]
			title_links_list = list(set(title_links_list) - set(recent_titles))

			# Check for reward
			r = get_reward(title)
			if r > 0 or len(title_links_list) == 0:
				is_terminal = 1
			rewards += r
			terminal_states += is_terminal

			# Save results and continue
			save_results_list(results_file, [title, r, is_terminal])
			if steps % print_freq == 0:
				print("\nCrawled {} pages, {} rewards, {} terminal_states".format(steps, 
					rewards, terminal_states))
			if is_terminal == 1 or steps_before_term >= term_steps:
				break
			
			next_title = wiki_crawl.choose_random_title(title_links_list)
			title, title_links_list = wiki_crawl.get_wiki_links(False, next_title)

	print("\nCrawled {} pages, {} rewards, {} terminal_states".format(steps, rewards, terminal_states))

if __name__ == "__main__":
	main()