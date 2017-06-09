# -*- coding: utf-8 -*-
import urllib.request
import os
import sys
import re
import csv
import time
import random
import ahocorasick
import numpy as np
import pandas as pd

from socket import timeout
from collections import OrderedDict


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


def check_strings(A, search_list, pagetxt):
	"""Use Aho Corasick algorithm to produce boolean list indicating
	prescence of strings within a longer string"""
	index_list = []
	for item in A.iter(pagetxt):
		index_list.append(item[1][0])

	output_list = np.array([0] * len(search_list))
	output_list[index_list] = 1
	return output_list.tolist()

def count_occurences(A, search_list, pagetxt):
	"""Use Aho Corasick algorithm to count number of strings present in longer string"""
	total_count = 0
	for item in A.iter(pagetxt):
		total_count += 1
	return(total_count)


def main():
	#---------------------- Load data
	# Load list of keywords, company names and URLs
	word_list = load_csv_to_list('../data/vert_desc_words.csv')
	name_list = load_csv_to_list('../data/company_names.csv')
	url_list = load_csv_to_list('../data/company_urls.csv')
	url_list = list(set(url_list))

	print(len(word_list))
	print(len(name_list))

	#---------------------- Search within pages
	# Build Aho Corasick search list for company names
	A_comp = ahocorasick.Automaton()
	for idx, s in enumerate(name_list):
		A_comp.add_word(s, (idx, s))
	A_comp.make_automaton()

	# Do the same for the vertical keywords
	A_vert = ahocorasick.Automaton()
	for idx, s in enumerate(word_list):
		A_vert.add_word(s, (idx, s))
	A_vert.make_automaton()

	# Build dict to store results
	results_dict = OrderedDict()
	results_dict['url'] = []
	results_dict['vert_desc_prop'] = []
	results_dict['vert_desc_count'] = []
	results_dict['name_prop'] = []
	results_dict['name_count'] = []
	results_dict['is_bad_url'] = []

	# Search through URLs and get proportions
	end_value = 10
	short_url_list = random.sample(url_list, end_value)

	for idx, url in enumerate(short_url_list):
		progress_bar(idx+1, end_value)
		try:
			pagetxt = urllib.request.urlopen(url, timeout=8).read().decode('utf-8').lower()
			results_dict['url'].append(url)
			results_dict['vert_desc_prop'].append(sum(check_strings(A_vert, word_list, pagetxt)) / len(word_list))
			results_dict['vert_desc_count'].append(count_occurences(A_vert, word_list, pagetxt))
			results_dict['name_prop'].append(sum(check_strings(A_comp, name_list, pagetxt)) / len(name_list))
			results_dict['name_count'].append(count_occurences(A_comp, name_list, pagetxt))
			results_dict['is_bad_url'].append(0)
		# except (UnicodeDecodeError, urllib.error.URLError, urllib.error.HTTPError, timeout):
		except:
			results_dict['url'].append(url)
			results_dict['vert_desc_prop'].append(None)
			results_dict['vert_desc_count'].append(None)
			results_dict['name_prop'].append(None)
			results_dict['name_count'].append(None)
			results_dict['is_bad_url'].append(1)
	print("")

	# Save results
	results_df = pd.DataFrame.from_dict(results_dict)
	print(results_df)
	# results_df.to_csv('../data/proportion_companies.csv', index=False)


if __name__ == "__main__":
	main()