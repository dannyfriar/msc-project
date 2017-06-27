import os
import sys
import re
import csv
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
storage = StorageEngine("/nvme/webcache/")

from nltk.corpus import stopwords, words, names
stops = stopwords.words("english")

def read_csv_to_list(filename):
	with open(filename) as f:  # relevant english words
		reader = csv.reader(f)
		csv_list = list(reader)
	csv_list = [c[0] for c in csv_list]
	return(csv_list)

##-------------------- Read in data
# Read in all URls, backlinks data and list of keywords
links_df = pd.read_csv('data/links_dataframe.csv')
url_list = links_df['url'].tolist()
url_list = [l.replace("http://", "").replace("https://", "") for l in url_list if type(l) is str if l[-4:] not in [".png", ".jpg", ".pdf", ".txt"]]
url_set = set(url_list)
backlinks = pd.read_csv('data/backlinks_clean.csv')
words_list = read_csv_to_list('data/word_feature_list.csv') + read_csv_to_list('data/domains_endings.csv')
words_list = list(set(words_list))
word_dict = dict(zip(words_list, list(range(len(words_list)))))
url_sample = url_list[:100]


count_vec = CountVectorizer(vocabulary=word_dict)
my_mat = count_vec.transform(url_sample).toarray()
print(my_mat.shape)
