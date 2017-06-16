# -*- coding: utf-8 -*-
import os
import sys
import re
import csv
import time
import random
import numpy as np
import pandas as pd

from evolutionai import StorageEngine
storage = StorageEngine("/nvme/webcache/")
# storage = StorageEngine("/nvme/links_db/")
# storage = StorageEngine("/companies-data/webcache2")


def progress_bar(value, endvalue, bar_length=20):
    """Print progress bar to the console"""
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

def get_list_of_links(url, s=storage):
	"""Use the LMDB database to get a list of links for a given URL"""
	page = s.get_page(url)
	if page is None:
		return []
	link_list = [l.url for l in page.links if l.url[:4] == "http"]
	return link_list

def get_all_links(url_list):
	"""Get all links from a list of URLs"""
	full_link_list = []
	for url in url_list:
		link_list = get_list_of_links(url)
		full_link_list = full_link_list + link_list
	full_link_list = full_link_list + url_list
	full_link_list = list(set(full_link_list))
	return full_link_list




def main():
	## Do for list of company URLs - companies from specific industry
	companies_df = pd.read_csv('../data/domains_clean.csv')
	companies_df = companies_df[companies_df['vert_code'] <= 69203]
	companies_df = companies_df[companies_df['vert_code'] >= 69101]
	# companies_df.to_csv('../data/domains_business_clean.csv', index=False)

	url_list = companies_df['url'].tolist()
	del companies_df

	## First hop links
	first_hop_links = get_all_links(url_list)
	first_hop_links = [l for l in first_hop_links if l not in url_list if l[0] != "\""]
	first_hop_links = [l.replace("http://", "") for l in first_hop_links]
	first_hop_links = [l.replace("https://", "") for l in first_hop_links]
	print(len(first_hop_links))
	first_hop_df = pd.DataFrame.from_dict({"url": first_hop_links})
	first_hop_df.to_csv("data/first_hop_links.csv", index=False, header=False)

	## Second hop links
	storage = StorageEngine("/companies-data/webcache2")
	# url = random.choice(first_hop_links)
	url = "www.scottfree.co.uk"
	print(url)
	print(storage.get_page(url))













if __name__ == "__main__":
	main()