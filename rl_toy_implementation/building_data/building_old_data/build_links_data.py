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
	skipped_urls = []
	for idx, url in enumerate(url_list):
		progress_bar(idx+1, len(url_list))
		try:
			link_list = get_list_of_links(url)
		except (UnicodeError, IndexError):
			skipped_urls.append(url)
		full_link_list = full_link_list + link_list
	full_link_list = full_link_list + url_list
	full_link_list = list(set(full_link_list))
	print("\nSkipped %d URLs" % len(skipped_urls))
	return full_link_list




def main():
	# ## Do for list of company URLs - companies from specific industry
	# companies_df = pd.read_csv('../data/domains_clean.csv')
	# companies_df = companies_df[companies_df['vert_code'] <= 69203]
	# companies_df = companies_df[companies_df['vert_code'] >= 69101]
	# # companies_df.to_csv('../data/domains_business_clean.csv', index=False)

	# url_list = companies_df['url'].tolist()
	# del companies_df

	# ## First hop links
	# first_hop_links = get_all_links(url_list)
	# first_hop_links = [l for l in first_hop_links if l not in url_list if l[0] != "\""]
	# first_hop_links = [l.replace("http://", "") for l in first_hop_links]
	# first_hop_links = [l.replace("https://", "") for l in first_hop_links]
	# first_hop_df = pd.DataFrame.from_dict({"url": first_hop_links})
	# # first_hop_df.to_csv("data/first_hop_links.csv", index=False, header=False)

	# ## Second hop links
	# print(len(first_hop_links))
	# second_hop_links = get_all_links(first_hop_links)
	# print(len(second_hop_links))
	# second_hop_links = [l for l in second_hop_links if l not in url_list+first_hop_links if len(l)>0 if l[0] != "\""]
	# second_hop_links = [l.replace("http://", "") for l in second_hop_links]
	# second_hop_links = [l.replace("https://", "") for l in second_hop_links]
	# second_hop_df = pd.DataFrame.from_dict({"url": second_hop_links})
	# # second_hop_df.to_csv("data/second_hop_links.csv", index=False, header=False)


	## Fourth hop links
	print("Reading previous data....")
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
	third_hop_links = third_hop_df['url'].tolist()
	url_list = url_list + third_hop_links
	url_list = set(url_list)
	del companies_df, first_hop_df, second_hop_df, third_hop_df

	## Third hop links
	print("Getting fourth hop links....")
	# third_hop_links = list(set(third_hop_links))
	print(len(third_hop_links))
	fourth_hop_links = get_all_links(third_hop_links)
	fourth_hop_links = [l for l in fourth_hop_links if len(l)>0 if l[0] != "\""]
	fourth_hop_links = list(set(fourth_hop_links) - url_list)
	print(len(fourth_hop_links))

	print("Replacing....")
	fourth_hop_links = [l.replace("http://", "").replace("https://", "") for l in fourth_hop_links]
	print("Saving data....")
	fourth_hop_df = pd.DataFrame.from_dict({"url": fourth_hop_links})
	fourth_hop_df.to_csv("data/fourth_hop_links.csv", index=False, header=False)







if __name__ == "__main__":
	main()