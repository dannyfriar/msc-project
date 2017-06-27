import os
import sys
import re
import csv
import time
import random
import pickle
import argparse
import ahocorasick
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import OrderedDict

from evolutionai import StorageEngine
storage = StorageEngine("/nvme/webcache/")

from nltk.corpus import stopwords, words, names
stops = stopwords.words("english")

##---------------------------------------------------------
def progress_bar(value, endvalue, bar_length=20):
    """Print progress bar to the console"""
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

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

def get_link_dict(url_list):
	"""Get all links from a list of URLs"""
	skipped_urls = []
	output_dict = OrderedDict()
	output_dict['back_url'] = []
	output_dict['url'] = []

	for idx, url in enumerate(url_list):
		progress_bar(idx+1, len(url_list))
		try:
			link_list = get_list_of_links(url)
		except (UnicodeError, IndexError):
			skipped_urls.append(url)
			link_list = []

		if len(link_list) > 0:
			output_dict['back_url'] += [url] * len(link_list)
			output_dict['url'] += link_list
		output_df = pd.DataFrame.from_dict(output_dict)

	print("\n")
	return output_df



def main():
	# Read in company URLs
	companies_df = pd.read_csv('../data/domains_clean.csv')
	companies_df = companies_df[companies_df['vert_code'] <= 69203]
	companies_df = companies_df[companies_df['vert_code'] >= 69101]
	reward_urls = companies_df['url'].tolist()
	reward_urls = [l.replace("http://", "").replace("https://", "") for l in reward_urls]

	# Read in first hop links
	with open("data/first_hop_links.csv") as f:  # relevant english words
		reader = csv.reader(f)
		first_hop_links = list(reader)
	first_hop_links = [l[0] for l in first_hop_links if len(l)>0]

	# Read in second hop links
	with open("data/second_hop_links.csv") as f:  # relevant english words
		reader = csv.reader(f)
		second_hop_links = list(reader)
	second_hop_links = [l[0] for l in second_hop_links if len(l)>0]

	# Read in third hop links
	with open("data/third_hop_links.csv") as f:  # relevant english words
		reader = csv.reader(f)
		third_hop_links = list(reader)
	third_hop_links = [l[0] for l in third_hop_links if len(l)>0]

	## Build the backlinks data
	company_backlinks = get_link_dict(reward_urls)
	first_backlinks = get_link_dict(first_hop_links)
	second_backlinks = get_link_dict(second_hop_links)
	all_backlinks = pd.concat([company_backlinks, first_backlinks, second_backlinks])
	# all_backlinks = company_backlinks
	all_backlinks.to_csv("data/backlinks.csv", index=False, header=True)
	print(len(all_backlinks))




if __name__ == main():
	main()