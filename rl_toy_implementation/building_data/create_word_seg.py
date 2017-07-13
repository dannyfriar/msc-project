import os
import re
import sys
import csv
import time
import string
import random
import numpy as np
import pandas as pd
import wordsegment


from wordsegment import segment
from collections import Counter
from urllib.parse import urlparse
from nltk.corpus import stopwords, words, names
stops = stopwords.words("english")

def progress_bar(value, endvalue, bar_length=20):
	percent = float(value) / endvalue
	arrow = '-' * int(round(percent * bar_length)-1) + '>'
	spaces = ' ' * (bar_length - len(arrow))
	sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
	sys.stdout.flush()

# # Old Read data
# links_df = pd.read_csv("../new_data/links_dataframe.csv")
# reward_urls = links_df[links_df['type']=='company-url']['url']
# reward_urls = [l.replace("www.", "") for l in reward_urls]
# url_list = list(set(links_df['url'].tolist()))
# # url_list = random.sample(url_list, 30000)
# url_list = list(set(url_list+reward_urls))

# # Read in links dataframe
# links_df = pd.read_csv("../new_data/links_dataframe.csv")
# rm_list = ['aarp.org', 'akc.org', 'alcon.com', 'lincoln.com', 'orlakiely.com', 
# 'red.com', 'ef.com', 'ozarksfirst.com']
# links_df['domain'] = links_df.domain.str.replace("www.", "")
# links_df = links_df[~links_df['domain'].isin(rm_list)]
# url_list = list(set(links_df['url'].tolist()))
# url_list = random.sample(url_list, 100000)

# ##-------------- Use word segmentation algorithm
# # Actually create the word segments - takes approx 100 minutes (faster with list comp?)
# t0 = time.time(); word_list = []
# for idx, url in enumerate(url_list):
# 	progress_bar(idx+1, len(url_list))
# 	if idx % 3000 == 0:
# 		pd.DataFrame.from_dict({"word":word_list}).to_csv("../new_data/new_sample_segmented_words.csv", mode='a', header=True, index=False)
# 		word_list = []
# 		print("\nRun for {} URLs in time {}".format(idx, time.time()-t0))
# 	word_list += segment(url)
# print("\n")

# Load in word segmentation full
with open("../data/segmented_words.csv") as f:  # relevant english words
	reader = csv.reader(f)
	word_list = list(reader)
word_list = [w[0] for w in word_list if len(w[0])>1]
with open("../new_data/new_sample_segmented_words.csv") as f:  # relevant english words
	reader = csv.reader(f)
	word_list_new = list(reader)
word_list_new = [w[0] for w in word_list_new if len(w[0])>1]
word_list = word_list + word_list_new


word_counts = pd.DataFrame.from_dict(Counter(word_list), orient='index').reset_index()
word_counts.columns = ['word','count']
word_counts = word_counts[word_counts['count']>1]
word_counts = word_counts[~word_counts['word'].isin(stops)]
word_counts = word_counts.sort_values(by="count", ascending=False).reset_index()
word_counts = word_counts[word_counts['count'] >= 50]  # Leaving 4.5k words
print(len(word_counts))
print(sum(word_counts['count']) / len(word_list))
# word_counts.to_csv("../data/segmented_words_df.csv", header=True, index=False)
word_counts.to_csv("../data/new_segmented_words_df.csv", header=True, index=False)


##-------------- Split on punctuation
# t0 = time.time(); word_list = []
# for idx, url in enumerate(url_list):
# 	progress_bar(idx+1, len(url_list))
# 	if idx % 100000 == 0:
# 		pd.DataFrame.from_dict({"word":word_list}).to_csv("../data/punc_split_words.csv", header=True, index=False)
# 		print("\nRun for {} URLs in time {}".format(idx, time.time()-t0))
# 	word_list += re.split("["+string.punctuation+"]+", url)

# # Load in punc split words
# with open("../data/punc_split_words.csv") as f:  # relevant english words
# 	reader = csv.reader(f)
# 	word_list = list(reader)

# word_list = [w[0] for w in word_list if len(w) > 0]
# word_list = [w for w in word_list if len(w) > 1]
# word_counts = pd.DataFrame.from_dict(Counter(word_list), orient='index').reset_index()
# word_counts.columns = ['word','count']
# word_counts = word_counts[word_counts['count']>1]
# word_counts = word_counts[~word_counts['word'].isin(stops)]
# word_counts = word_counts.sort_values(by="count", ascending=False).reset_index()
# word_counts = word_counts[word_counts['count'] >= 30]  # Leaving 4k words
# word_counts.to_csv("../data/punc_split_words_df.csv", header=True, index=False)




