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
from collections import OrderedDict
from bs4 import BeautifulSoup


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


def make_automaton(string_list):
	"""Make Aho-Corasick automaton from a list of strings"""
	A = ahocorasick.Automaton()
	for idx, s in enumerate(string_list):
		A.add_word(s, (idx, s))
	return A.make_automaton()

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
	word_list = load_csv_to_list('../data/vert_desc_words.csv')
	name_list = load_csv_to_list('../data/company_names.csv')
	url_list = list(set(load_csv_to_list('../data/company_urls.csv')))
	# print(find_links(start_url))

	# Make Aho-Corasick automatons	
	A_vert = make_automaton(word_list)
	A_comp = make_automaton(name_list)

	##-------------- Crawling ---------------
	pages_crawled = 0
	rewards = 0

	stop

	while True:
		# Choose start URL
		url = random.choice(url_list)

		while True:
			print(url)
			pages_crawled += 1
			print("------ Crawled %d pages, receiving %d rewards" % (pages_crawled, rewards))

			# Follow random link from the page
			link_list = find_links(url)
			if len(link_list) == 0:
				print("Choosing again...")
				break
			url = random.choice(link_list)

			# Check for reward within page
			try:
				pagetxt = urllib.request.urlopen(url, timeout=8).read().decode('utf-8').lower()
				if sum(check_strings(A_vert, word_list, pagetxt)) > 0:
					rewards += 1
			except:
				pass


if __name__ == "__main__":
	random.seed(0)
	main()