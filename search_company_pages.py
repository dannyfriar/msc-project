# -*- coding: utf-8 -*-
import urllib.request
import os
import sys
import re
import csv
import time
import ahocorasick
import numpy as np
import pandas as pd

from socket import timeout


def progress_bar(value, endvalue, bar_length=20):
    """Print progress bar to the console"""
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


def check_strings(A, search_list, pagetxt):
	"""Use Aho Corasick algorithm to produce boolean list indicating
	prescence of strings within a longer string"""
	index_list = []
	for item in A.iter(pagetxt):
		index_list.append(item[1][0])

	output_list = np.array([0] * len(search_list))
	output_list[index_list] = 1
	return output_list.tolist()


def main():
	# Load list of keywords
	with open('data/vert_desc_words.csv', 'r') as f:
		reader = csv.reader(f)
		word_list = list(reader)[0]

	# Build Aho Corasick search list
	A = ahocorasick.Automaton()
	for idx, s in enumerate(word_list):
		A.add_word(s, (idx, s))
	A.make_automaton()

	# Load list of URLs
	with open('data/company_urls.csv', 'r') as f:
		reader = csv.reader(f)
		url_list = list(reader)[0]

	url_list = list(set(url_list))

	# Search through URLs and get proportions
	results_list = []
	bad_url_list = []
	end_value = 1000

	for idx, url in enumerate(url_list[:end_value]):
		progress_bar(idx+1, end_value)
		try:
			pagetxt = urllib.request.urlopen(url, timeout=10).read().decode('utf-8').lower()
			proportion = sum(check_strings(A, word_list, pagetxt)) / len(word_list)
			results_list.append(proportion)
		except (UnicodeDecodeError, urllib.error.URLError, urllib.error.HTTPError, timeout):
			results_list.append(None)
			bad_url_list.append(url)
	print("")

	results_list = list(filter(None.__ne__, results_list))
	print("Mean proportion = %f" % np.mean(np.array(results_list)))



if __name__ == "__main__":
	main()