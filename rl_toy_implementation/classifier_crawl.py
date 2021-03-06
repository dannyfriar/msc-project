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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from evolutionai import StorageEngine
storage = StorageEngine("/nvme/rl_project_webcache/")

from nltk.corpus import stopwords, words, names
stops = stopwords.words("english")

RESULTS_FOLDER = "results/classifier_results/"
MODEL_FOLDER = "models/"

##-----------------------------------------------------------
##-------- Functions ----------------------------------------
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

def pad_shorten(s, max_len):
	"""Pad URLs to max length"""
	try:
		s_out = np.pad(s, (0, max_len-len(s)), 'constant', constant_values=(0,0))
	except ValueError as e:
		print("In exception!!")
		s_out = s[:max_len]
	return s_out

def build_url_feature_matrix(count_vec, url_list, embeddings, max_len):
	"""Return 2d numpy array of booleans"""
	feature_matrix = count_vec.transform(url_list).toarray()
	if len(url_list) == 1:
		s = np.nonzero(feature_matrix)[1]
		s = pad_shorten(s, max_len)
		# s = np.pad(s, (0, max_len-len(s)), 'constant', constant_values=(0,0))
		return s.reshape(1, -1)
	else:
		s_prime = np.nonzero(feature_matrix)
		splits = np.cumsum(np.bincount(s_prime[0]))
		s_prime = np.split(s_prime[1], splits.tolist())[:-1]
		s_prime = [pad_shorten(s, max_len) for s in s_prime]
		s_prime = np.stack(s_prime, axis=0)
		return s_prime

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
	except (UnicodeError, ValueError):
		return []
	try:
		link_list = [l.url.replace("http://", "").replace("https://", "") for l in page.links if l.url[:4] == "http"]
		link_list = link_list + [l.replace("www.", "") for l in link_list]
	except UnicodeDecodeError:
		return []
	return link_list

##-----------------------------------------------------------
##-------- RL Functions -------------------------------------
def get_random_url(url_list, recent_urls):
	"""Get random url that is not in list of recent URLs"""
	url = random.choice(url_list)
	while url in recent_urls:
		url = random.choice(url_list)
	return url

def get_reward(url, A_company, company_urls):
	"""Return 1 if company URL, 0 otherwise"""
	idx_list = check_strings(A_company, company_urls, url)
	if sum(idx_list) > 0:
		reward_url_idx = np.nonzero(idx_list)[0][0]
		return 1, reward_url_idx
	return 0, None

##-----------------------------------------------------------
##-------- Classifier ---------------------------------------
class CNNClassifier(object):
	def __init__(self, filter_sizes, num_filters, embeddings, embedding_size, max_len, tf_model_folder):
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters
		self.embeddings = embeddings
		self.embedding_size = embedding_size
		self.max_len = max_len
		self.num_filters_total = self.num_filters * len(self.filter_sizes)

		# Set up training parameters and TF placeholders
		self.x = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len], name='x')
		self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')
		self.embeddings_net()
		self.tf_model_folder = tf_model_folder

	def embeddings_net(self):
		embedded_x = tf.nn.embedding_lookup(self.embeddings, self.x)
		embedded_x = tf.expand_dims(embedded_x, -1)
		embedded_x = tf.nn.l2_normalize(embedded_x, 1)

		W_out = tf.get_variable("W_out", [self.num_filters_total, 1], 
			initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001))
		b_out = tf.get_variable("b_out", [1], initializer=tf.constant_initializer(0.001))

		# Convolutions for s
		pooled_outputs = []
		for i, filter_size in enumerate(self.filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
				b = tf.Variable(tf.constant(0.01, shape=[self.num_filters]), name="b")
				conv = tf.nn.conv2d(tf.cast(embedded_x, tf.float32), W, 
					strides=[1, 1, 1, 1], padding="VALID", name="conv")
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				pooled = tf.nn.max_pool(h, ksize=[1, self.max_len-filter_size+1, 1, 1],
					strides=[1, 1, 1, 1], padding='VALID', name="pool")
				pooled_outputs.append(pooled)

		# Fully connected for s
		h_pool = tf.concat(pooled_outputs, 3)
		h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])
		self.x_out = tf.nn.sigmoid(tf.matmul(h_pool_flat, W_out) + b_out)

##-----------------------------------------------------------
##-----------------------------------------------------------
def main():
	##-------------------- Parameters
	cycle_freq = 50
	term_steps = 50
	num_steps = 200000  # no. crawled pages before stopping
	print_freq = 100000
	max_len = 50
	embedding_size = 300
	filter_sizes = [1, 2, 3, 4]
	num_filters = 4
	n_runs = 5

	##-------------------- Read in data
	print("Loading Data...")
	links_df = pd.read_csv("new_data/links_dataframe.csv")
	rm_list = ['aarp.org', 'akc.org', 'alcon.com', 'lincoln.com', 'orlakiely.com', 
	'red.com', 'ef.com', 'ozarksfirst.com']  # remove mis-labelled reward URLs
	links_df['domain'] = links_df.domain.str.replace("www.", "")
	links_df = links_df[~links_df['domain'].isin(rm_list)]
	reward_urls = links_df[links_df['type']=='company-url']['url']
	reward_urls = [l.replace("www.", "") for l in reward_urls]
	reward_urls = [l for l in reward_urls if l not in rm_list]
	A_company = init_automaton(reward_urls)  # Aho-corasick automaton
	A_company.make_automaton()
	url_set = set(links_df['url'].tolist())
	url_list = list(url_set)

	# Embeddings matrix
	print("Loading words and embeddings...")
	embeddings = np.loadtxt('../../url_embeddings_matrix.csv', delimiter=',')

	# URL words
	words_list = pd.read_csv("../../embedded_url_word_list.csv")['word'].tolist()
	word_dict = dict(zip(words_list, list(range(len(words_list)))))
	count_vec = CountVectorizer(vocabulary=word_dict)

	# File locations
	all_urls_file = RESULTS_FOLDER + "all_urls_crawler.csv"
	model_save_file = MODEL_FOLDER + "classifier"

	##------------------- Initialize TF graph/session
	if os.path.isfile(all_urls_file):
		os.remove(all_urls_file)

	for run in range(n_runs):
		print("#------------- Run {}".format(run+1))
		step_count = 0; pages_crawled = 0; total_reward = 0; terminal_states = 0
		recent_urls = []; reward_pages = []; found_rewards = []; reward_domain_set = set()

		tf.reset_default_graph()
		agent = CNNClassifier(filter_sizes, num_filters, embeddings, embedding_size, max_len, model_save_file)
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		with tf.Session() as sess:
			sess.run(init)
			print("Reloading model...")
			saver = tf.train.import_meta_graph(model_save_file+"/tf_model.meta")
			saver.restore(sess, tf.train.latest_checkpoint(model_save_file))
			all_vars = tf.get_collection('vars')

			while step_count < num_steps:
				url = get_random_url(url_list, recent_urls)
				steps_without_terminating = 0

				while step_count < num_steps:
					step_count += 1

					# Keep track of recent URLs (to avoid loops)
					recent_urls.append(url)
					if len(recent_urls) > cycle_freq:
						recent_urls = recent_urls[-cycle_freq:]

					# Get rewards and remove domain if reward
					r, reward_url_idx = get_reward(url, A_company, reward_urls)
					pages_crawled += 1
					total_reward += r

					link_list = get_list_of_links(url)
					link_list = set(link_list).intersection(url_set)
					link_list = list(link_list - set(recent_urls))

					# Check if terminal state
					if r > 0 or len(link_list) == 0:
						terminal_states += 1
						is_terminal = 1
						next_state_array = np.zeros(shape=(1, max_len))  # doesn't matter what this is
					else:
						is_terminal = 0
						steps_without_terminating += 1
						next_state_array = build_url_feature_matrix(count_vec, link_list, embeddings, max_len)

					v_next = sess.run(agent.x_out, feed_dict={agent.x: next_state_array})
					v_next = v_next.reshape(-1)

					# Print progress + save transitions
					progress_bar(step_count+1, num_steps)
					if step_count % print_freq == 0:
						print("\nCrawled {} pages, total reward = {}, # terminal states = {}, remaining rewards = {}"\
						.format(pages_crawled, total_reward, terminal_states, len(reward_urls)))

					with open(all_urls_file, "a") as csv_file:
						writer = csv.writer(csv_file, delimiter=',')
						writer.writerow([url, r, is_terminal, run])

					# Choose next URL (and check for looping)
					if is_terminal == 1:
						break
					if steps_without_terminating >= term_steps:  # to prevent cycles
						break
					a = np.argmax(v_next)
					url = link_list[a]

		sess.close()

if __name__ == "__main__":
	main()