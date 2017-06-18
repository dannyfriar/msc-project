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

from evolutionai import StorageEngine
storage = StorageEngine("/nvme/webcache/")

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


def main():
	##-- Read in data
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


	##-- Check which URL contains the most links
	most_links_num = 0
	page_with_most_links = random.choice(url_list)

	for idx, url in enumerate(url_list):
		progress_bar(idx+1, len(url_list))
		link_list = get_list_of_links(url)
		link_list = [l for l in link_list if l in url_list]
		if len(link_list) > most_links_num:
			most_links_num = len(link_list)
			page_with_most_links = url

	print("\nPage with most links is {}, with {}".format(page_with_most_links, most_links_num))






if __name__ == "__main__":
	main()