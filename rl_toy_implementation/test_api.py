# -*- coding: utf-8 -*-
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

from evolutionai import StorageEngine
storage = StorageEngine("/nvme/links_db/")

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
##-------- URL API functions --------------------------------
def get_list_of_links(url, s=storage):
	"""Use the LMDB database to get a list of links for a given URL"""
	page = s.get_page(url)
	link_list = [l.url for l in page.links if l.url[:4] == "http"]
	return link_list


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
	
	link_list = get_list_of_links("http://evolutionleader.com/contact-2/")
	print(link_list)


















if __name__ == "__main__":
	main()





