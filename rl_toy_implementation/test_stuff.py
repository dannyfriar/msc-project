import random
import numpy as np
import pandas as pd

from evolutionai import StorageEngine
storage = StorageEngine("/nvme/rl_project_webcache/")


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
			return [], []
	except UnicodeError:
		return [], []
	try:
		link_list = [l.url.replace("http://", "").replace("https://", "") for l in page.links if l.url[:4] == "http"]
		link_list = link_list + [l.replace("www.", "") for l in link_list]
		text_list = [l.text for l in page.links if l.url[:4] == "http"]
	except UnicodeDecodeError:
		return [], []
	return link_list, text_list


links_df = pd.read_csv("new_data/links_dataframe.csv")
reward_urls = links_df[links_df['type']=='company-url']['url']
reward_urls = [l.replace("www.", "") for l in reward_urls]
url_set = set(links_df['url'].tolist())
url_list = list(url_set)


url = random.choice(url_list)
url_list, text_list = get_list_of_links(url)
print(url_list)
print(text_list)

