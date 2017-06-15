import os
import sys
import re
import csv
import time
import random
import ahocorasick
import numpy as np
import pandas as pd
import tensorflow as tf

from nltk.corpus import stopwords, words, names
stops = stopwords.words("english")

##-----------------------------------------------------------
##-------- Miscellaneous Functions --------------------------
def progress_bar(value, endvalue, bar_length=20):
    """Print progress bar to the console"""
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

def load_csv_to_list(file_name):
	"""Loads single column CSV as list"""
	with open(file_name) as f:
		reader = csv.reader(f)
		return list(reader)[0]

##-----------------------------------------------------------
##-------- String Functions ---------------------------------
def get_words_from_url(url, pattern=re.compile(r'[\:/?=\-&.]+',re.UNICODE)):
	"""Split URL into words"""
	word_list = pattern.split(url)
	word_list = list(set(word_list))
	word_list = [word for word in word_list if word not in ['http', 'www']+stops]
	return word_list

def bow_url_list(url_list):
	"""Takes URL list and forms bag-of-words representation"""
	all_urls = "-".join(url_list)
	return get_words_from_url(all_urls)

def init_automaton(string_list):
	"""Make Aho-Corasick automaton from a list of strings"""
	A = ahocorasick.Automaton()
	for idx, s in enumerate(string_list):
		A.add_word(s, (idx, s))
	return A

def check_strings(A, search_list, string_to_search):
	"""Use Aho Corasick algorithm to produce boolean list indicating
	prescence of strings within a longer string"""
	index_list = []
	for item in A.iter(string_to_search):
		index_list.append(item[1][0])

	output_list = np.array([0] * len(search_list))
	output_list[index_list] = 1
	return output_list.tolist()


def main():

	## REGEX PATTERN AND STOPWORDS
	pattern = re.compile(r'[\:/?=\-&.]+',re.UNICODE)

	# ## Test example - single URL
	# url = "http://www.bbc.co.uk/news"
	# words = get_words_from_url(url, pattern)
	
	## Do for list of company URLs - companies from specific industry
	companies_df = pd.read_csv('../data/domains_clean.csv')
	companies_df = companies_df[companies_df['vert_code'] <= 69203]
	companies_df = companies_df[companies_df['vert_code'] >= 69101]
	url_list = companies_df['url'].tolist()
	del companies_df

	## Import english words and build Aho-Corasick automaton
	english_words = words.words() + names.words()
	english_words = [w for w in english_words if w not in stops]
	english_words = [w for w in english_words if len(w) > 1]
	print(len(english_words))
	A = init_automaton(english_words)
	A.make_automaton()

	# Test URL
	url = random.choice(url_list)
	print(url)
	bin_string = check_strings(A, english_words, url)
	nonzero_indices = list(np.nonzero(bin_string)[0])
	present_words = [english_words[i] for i in nonzero_indices]
	print(present_words)





if __name__ == "__main__":
	main()