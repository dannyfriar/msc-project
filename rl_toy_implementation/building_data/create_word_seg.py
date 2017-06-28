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
from nltk.corpus import stopwords, words, names
stops = stopwords.words("english")

def progress_bar(value, endvalue, bar_length=20):
	percent = float(value) / endvalue
	arrow = '-' * int(round(percent * bar_length)-1) + '>'
	spaces = ' ' * (bar_length - len(arrow))
	sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
	sys.stdout.flush()

# Read data
links_df = pd.read_csv('../data/links_dataframe.csv')
url_list = links_df['url'].tolist()
url_list = [l.replace("http://", "").replace("https://", "") for l in url_list if type(l) is str if l[-4:] not in [".png", ".jpg", ".pdf", ".txt"]]

##-------------- Use word segmentation algorithm
# Actually create the word segments - takes approx 100 minutes (faster with list comp?)
# t0 = time.time(); word_list = []
# for idx, url in enumerate(url_list):
# 	progress_bar(idx+1, len(url_list))
# 	if idx % 3000 == 0:
# 		# word_list = list(set(word_list))
# 		pd.DataFrame.from_dict({"word":word_list}).to_csv("../data/segmented_words.csv", header=True, index=False)
# 		print("\nRun for {} URLs in time {}".format(idx, time.time()-t0))
# 	word_list += segment(url)
# print("\n")

# # Load in word segmentation full
# with open("../data/segmented_words.csv") as f:  # relevant english words
# 	reader = csv.reader(f)
# 	word_list = list(reader)

# word_list = [w[0] for w in word_list if len(w[0])>1]
# word_counts = pd.DataFrame.from_dict(Counter(word_list), orient='index').reset_index()
# word_counts.columns = ['word','count']
# word_counts = word_counts[word_counts['count']>1]
# word_counts = word_counts[~word_counts['word'].isin(stops)]
# word_counts = word_counts.sort_values(by="count", ascending=False).reset_index()
# # print(word_counts.head(n=10))
# word_counts = word_counts[word_counts['count'] >= 50]  # Leaving 4.5k words
# word_counts.to_csv("../data/segmented_words_df.csv", header=True, index=False)


##-------------- Split on punctuation
# t0 = time.time(); word_list = []
# for idx, url in enumerate(url_list):
# 	progress_bar(idx+1, len(url_list))
# 	if idx % 100000 == 0:
# 		pd.DataFrame.from_dict({"word":word_list}).to_csv("../data/punc_split_words.csv", header=True, index=False)
# 		print("\nRun for {} URLs in time {}".format(idx, time.time()-t0))
# 	word_list += re.split("["+string.punctuation+"]+", url)
# print("\n")

# Load in punc split words
with open("../data/punc_split_words.csv") as f:  # relevant english words
	reader = csv.reader(f)
	word_list = list(reader)

word_list = [w[0] for w in word_list if len(w) > 0]
word_list = [w for w in word_list if len(w) > 1]
word_counts = pd.DataFrame.from_dict(Counter(word_list), orient='index').reset_index()
word_counts.columns = ['word','count']
word_counts = word_counts[word_counts['count']>1]
word_counts = word_counts[~word_counts['word'].isin(stops)]
word_counts = word_counts.sort_values(by="count", ascending=False).reset_index()
# word_counts = word_counts[word_counts['count'] >= 30]  # Leaving k words
word_counts.to_csv("../data/punc_split_words.csv", header=True, index=False)




