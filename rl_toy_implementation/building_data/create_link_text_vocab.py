import time
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

from evolutionai import StorageEngine
storage = StorageEngine("/nvme/rl_project_webcache/")

from nltk.corpus import stopwords, words, names
stops_set = set(stopwords.words("english"))
words_set = set(words.words())


def progress_bar(value, endvalue, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


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
	except (UnicodeError, ValueError):
		return []
	try:
		text_list = [l.text.lower() for l in page.links if l.url[:4] == "http"]
	except UnicodeDecodeError:
		return []
	return text_list


links_df = pd.read_csv("new_data/links_dataframe.csv")
reward_urls = links_df[links_df['type']=='company-url']['url']
reward_urls = [l.replace("www.", "") for l in reward_urls]
url_set = set(links_df['url'].tolist())
url_list = list(url_set)
all_text = []

# # Produce text link data
# for idx, url in enumerate(url_list):
# 	progress_bar(idx+1, len(url_list))
# 	if idx % 50000 == 0:
# 		pd.DataFrame.from_dict({"link_text":all_text}).to_csv("new_data/link_text.csv", mode='a', header=True, index=False)
# 		print("\nRun for {} words.".format(idx+1))
# 		all_text = []
# 	text_list = get_list_of_links(url)
# 	if len(text_list) > 0 and type(text_list) != tuple:
# 		text_list = set(" ".join(text_list).split())
# 		text_list = list(text_list.intersection(words_set) - stops_set)
# 		all_text += text_list


# # Read in text link data and write all words
# print("Reading data and splitting words...")
# all_text = pd.read_csv("new_data/link_text.csv")['link_text'].tolist()
# print(len(all_text))
# word_counts = pd.DataFrame.from_dict(Counter(all_text), orient='index').reset_index()
# word_counts.to_csv("new_data/link_text_word_count.csv", header=True, index=False)


# Read in these words and look at frequency distribution etc...
word_counts = pd.read_csv("new_data/link_text_word_count.csv", names=["word", "count"])
word_counts = word_counts.sort_values(by="count", ascending=False).reset_index()
full_length = sum(word_counts['count'])
word_counts = word_counts[word_counts['count'] >= 5000]
print(len(word_counts['word'].tolist()))
print(sum(word_counts['count']) / full_length)
word_counts.to_csv("new_data/link_text_vocab.csv", header=True, index=False)

# print(word_counts.head(n=10))
















