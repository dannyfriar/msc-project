import os
import sys
import re
import csv
import pdb
import time
import random
import pickle
import argparse
import ahocorasick
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer

from evolutionai import StorageEngine
storage = StorageEngine("/nvme/rl_project_webcache/")

from nltk.corpus import stopwords, words, names
stops = stopwords.words("english")



def build_url_feature_matrix(count_vec, url_list):
	"""Return 2d numpy array of booleans"""
	feature_matrix = count_vec.transform(url_list).toarray()
	return feature_matrix


# Read in word list
words_list = pd.read_csv("data/segmented_words_df.csv")['word'].tolist()
word_dict = dict(zip(words_list, list(range(len(words_list)))))
count_vec = CountVectorizer(vocabulary=word_dict)

# Read in links dataframe
links_df = pd.read_csv("new_data/links_dataframe.csv")
rm_list = ['aarp.org', 'akc.org', 'alcon.com', 'lincoln.com', 'orlakiely.com', 
'red.com', 'ef.com', 'ozarksfirst.com']
links_df['domain'] = links_df.domain.str.replace("www.", "")
links_df = links_df[~links_df['domain'].isin(rm_list)]
url_list = list(set(links_df['url'].tolist()))

for i in range(10):
	url = random.choice(url_list)
	print(url)
	page = storage.get_page(url)
	if page is not None:
		print("Title: {}".format(page.title))
	print(np.sum(build_url_feature_matrix(count_vec, [url])))
	input("Press enter")







