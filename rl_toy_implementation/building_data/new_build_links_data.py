import sys
import lmdb
import string
import random
import ahocorasick
import numpy as np
import pandas as pd

from urllib.parse import urlparse
from evolutionai import StorageEngine
from sklearn.feature_extraction.text import CountVectorizer

DB_PATH = "/nvme/webcache1/"
storage = StorageEngine(DB_PATH)

##------------------------------------------------------
def progress_bar(value, endvalue, bar_length=20):
	percent = float(value) / endvalue
	arrow = '-' * int(round(percent * bar_length)-1) + '>'
	spaces = ' ' * (bar_length - len(arrow))
	sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
	sys.stdout.flush()

def init_automaton(string_list):
	"""Make Aho-Corasick automaton from a list of strings"""
	A = ahocorasick.Automaton()
	for idx, s in enumerate(string_list):
		A.add_word(s, (idx, s))
	return A

def check_strings(A, search_list, string_to_search):
	"""Aho Corasick algorithm, return boolean list of strings within longer string"""
	index_list = []
	for item in A.iter(string_to_search):
		index_list.append(item[1][0])

	output_list = np.array([0] * len(search_list))
	output_list[index_list] = 1
	return output_list.tolist()


def get_list_of_links(url, s=storage):
	"""Use the LMDB database to get a list of links for a given URL"""
	try:
		page = s.get_page(url)
		if page is None:
			page = s.get_page(url+"/")
		if page is None:
			page = s.get_page("www."+url)
		if page is None:
			page = s.get_page("www."+url+"/")
		if page is None:
			return []
	except UnicodeError:
		return []
	try:
		# link_list = [l.url.replace("http://", "").replace("https://", "") for l in page.links if ".uk" in l.url and l.url[:4] == "http"]
		link_list = [l.url.replace("http://", "").replace("https://", "") for l in page.links if l.url[:4] == "http"]
	except UnicodeDecodeError:
		return []
	return link_list

def main():
	env = lmdb.open(DB_PATH, map_size=1024**4)
	with env.begin() as txn:
		link_list = [key.decode('utf-8') for key, _ in txn.cursor()]
	link_list = list(set(link_list))
	print("Number of unique page links {}".format(len(link_list)))

	domain_list = [urlparse(l).netloc.replace("www.", "") for l in link_list]
	domain_list = list(set(domain_list))
	print("Number of unique domains {}".format(len(link_list)))

	links_df = pd.read_csv('../data/links_dataframe.csv')
	reward_urls = [l.replace("www.", "") for l in links_df[links_df['hops']==0]['url'].tolist()]
	A_company = init_automaton(reward_urls) # Aho-corasick automaton for companies
	A_company.make_automaton()

	# # Check random link actually links to company page
	# random_links = " ".join(get_list_of_links(random.choice(link_list)))
	# print(sum(check_strings(A_company, reward_urls, random_links)))

	print("Getting all links...")
	all_urls = []
	for idx, link in enumerate(link_list):
		progress_bar(idx+1, len(link_list))
		all_urls += get_list_of_links(link)
	all_urls+= link_list
	all_urls = list(set(all_urls))
	print("\nNumber of unique links {}".format(len(all_urls)))
	all_urls = " ".join(all_urls)

	pct_rewards = np.mean(np.array(check_strings(A_company, reward_urls, all_urls)))
	print("% Reward URLs found = {}".format(pct_rewards))






if __name__ == "__main__":
	main()