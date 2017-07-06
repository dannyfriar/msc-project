import sys
import pdb
import lmdb
import string
import random
import ahocorasick
import numpy as np
import pandas as pd

from operator import itemgetter
from urllib.parse import urlparse
from evolutionai import StorageEngine
from sklearn.feature_extraction.text import CountVectorizer

DB_PATH = "/nvme/webcache1/"
# DB_PATH = "/nvme/second_level/"
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
	# env = lmdb.open(DB_PATH, map_size=1024**4)
	# with env.begin() as txn:
	# 	link_list = [key.decode('utf-8') for key, _ in txn.cursor()]
	# link_list = list(set(link_list))
	# print("Number of unique page links {}".format(len(link_list)))

	# # clean_link_list = [l.replace("http://", "").replace("https://", "") for l in link_list]
	# # clean_link_list = list(set(clean_link_list))
	# # pd.DataFrame.from_dict({"url": clean_link_list}).to_csv("../new_data/first_hop_links.csv", index=False)

	# domain_list = [urlparse(l).netloc.replace("www.", "") for l in link_list]
	# domain_list = list(set(domain_list))
	# print("Number of unique domains {}".format(len(domain_list)))

	# # Company URLs (3,000)
	# links_df = pd.read_csv('../data/links_dataframe.csv')
	# reward_urls = [l.replace("www.", "") for l in links_df[links_df['hops']==0]['url'].tolist()]
	# A_company = init_automaton(reward_urls) # Aho-corasick automaton for companies
	# A_company.make_automaton()

	# # # First hop pages (read in first hop links)
	# # links_df = pd.read_csv('../new_data/first_hop_links.csv')
	# # reward_urls = [l.replace("www.", "") for l in links_df['url'].tolist()]
	# # A_company = init_automaton(reward_urls) # Aho-corasick automaton for first hop pages
	# # A_company.make_automaton()
	# # reward_set = set(reward_urls)


	# # # Check random link actually links to company page
	# # random_links = " ".join(get_list_of_links(random.choice(link_list)))
	# # print(sum(check_strings(A_company, reward_urls, random_links)))

	# # # Find proportion of links that are in the reward (company) domain
	# # total = 0
	# # for idx, link in enumerate(link_list):
	# # 	progress_bar(idx+1, len(link_list))
	# # 	if sum(check_strings(A_company, reward_urls, link)) >= 1:
	# # 		total += 1
	# # print("\n Percentage links in reward list {}".format(total/len(link_list)))

	# print("Getting all links...")
	# all_urls = []
	# # total = 0
	# # got = 0
	# for idx, link in enumerate(link_list):
	# 	progress_bar(idx+1, len(link_list))
	# 	temp_url_list = get_list_of_links(link)
	# 	# total += 1
	# 	# if len(set(temp_url_list).intersection(reward_set)) > 0:
	# 		# got += 1
	# 	all_urls += temp_url_list

	# # print("\n")
	# # print(got)
	# # print(total)
	# all_urls+= link_list
	# all_urls = list(set(all_urls))
	# print("\nNumber of unique links {}".format(len(all_urls)))
	# all_urls_string = " ".join(all_urls)

	# reward_indices = check_strings(A_company, reward_urls, all_urls_string)
	# pct_rewards = np.mean(np.array(reward_indices))
	# print("% Reward URLs found = {}".format(pct_rewards))

	# # Find all the reward pages in the company domains (as these need to be crawled)
	# actual_reward_pages = []
	# for idx, url in enumerate(all_urls):
	# 	progress_bar(idx+1, len(all_urls))
	# 	if sum(check_strings(A_company, reward_urls, url)) > 0:
	# 		actual_reward_pages.append(url)

	# pd.DataFrame.from_dict({'url':actual_reward_pages}).to_csv("../new_data/actual_reward_pages.csv", index=False)
	actual_reward_pages = pd.read_csv("../new_data/actual_reward_pages.csv")['url'].tolist()
	actual_reward_pages = [l.replace("http://", "").replace("https://", "") for l in actual_reward_pages]
	actual_reward_pages = list(set(actual_reward_pages))
	pd.DataFrame.from_dict({'url':actual_reward_pages}).to_csv("../new_data/actual_reward_pages.csv", index=False)


	# clean_all_urls = [l.replace("http://", "").replace("https://", "") for l in all_urls]
	# clean_all_urls = list(set(clean_all_urls))
	# pd.DataFrame.from_dict({"url": clean_all_urls}).to_csv("../new_data/first_hop_outgoing_uk_links.csv", index=False)

	# found_rewards = [reward_urls[i] for i in np.nonzero(reward_indices)[0]]
	# pd.DataFrame.from_dict({"url": found_rewards}).to_csv("../new_data/company_urls.csv", index=False)




if __name__ == "__main__":
	main()