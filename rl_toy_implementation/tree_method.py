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

RESULTS_FOLDER = "results/tree_results/"
MODEL_FOLDER = "models/"

##-----------------------------------------------------------
##-------- Miscellaneous Functions --------------------------
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
			return []
	except UnicodeError:
		return []
	try:
		link_list = [l.url.replace("http://", "").replace("https://", "") for l in page.links if l.url[:4] == "http"]
	except UnicodeDecodeError:
		return []
	return link_list

def lookup_domain_name(links_df, domain_url):
	"""Returns list of all URLs within domain web site (in database)"""
	return links_df[links_df['domain'].values == domain_url]['url'].tolist()

def append_backlinks(url, backlinks, link_list):
	"""Get the backlink for the URL, returns a string"""
	backlink =  backlinks[backlinks['url'].values == url]['back_url'].tolist()
	if len(backlink) == 0:
		return link_list
	link_list.append(backlink[0])
	return link_list


##-----------------------------------------------------------
##-------- String Functions ---------------------------------
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

def build_url_feature_matrix(word_dict, url_list, revisit, found_rewards):
	"""Return 2d numpy array of booleans"""
	count_vec = CountVectorizer(vocabulary=word_dict)
	feature_matrix = count_vec.transform(url_list).toarray()
	if revisit == True:
		return feature_matrix
	extra_vector = np.array([1 if l in found_rewards else 0 for l in url_list]).reshape(-1, 1)
	feature_matrix = np.concatenate((feature_matrix, extra_vector), axis=1)
	return feature_matrix


##-----------------------------------------------------------
##-------- RL Functions -------------------------------------
def get_reward(url, A_company, company_urls):
	"""Return 1 if company URL, 0 otherwise"""
	if sum(check_strings(A_company, company_urls, url)) > 0:
		return 1
	return 0

def epsilon_greedy(epsilon, action_list):
	"""Returns index of chosen action"""
	if random.uniform(0, 1) > epsilon:
		return np.argmax(action_list)
	else:
		return random.randint(0, len(action_list)-1)

def compute_value(url, A_company, company_urls):
	"""Returns the value of a given URL"""
	state_list = [url]; reward_list = []
	r = get_reward(url, A_company, company_urls)
	reward_list.append(r)
	if r > 0:
		return state_list, reward_list

	# First hop
	first_hop_links = get_list_of_links(url)
	first_hop_links = append_backlinks(url, backlinks, first_hop_links)
	if len(first_hop_links) == 0:
		reward_list.append(0)
		return state_list, reward_list

	first_hop_rewards = [get_reward(l, A_company, company_urls) for l in first_hop_links]
	if sum(first_hop_links) > 0:
		state_list.append(first_hop_links[np.argmax(first_hop_rewards)])
		reward_list.append(gamma)
		return state_list, reward_list

	# Second hop
	second_hop_links = get_all_links(first_hop_links)
	backlink_list = backlinks[backlinks['url'].isin(first_hop_links)]['back_url'].tolist()
	second_hop_links += backlink_list
	if len(second_hop_links) == 0:
		reward_list.append(0)
		return state_list, reward_list

	second_hop_rewards = [get_reward(l, A_company, company_urls) for l in first_hop_links]
	if r 0:
		return gamma**2

	# Third hop
	third_hop_links = get_all_links(second_hop_links)
	backlink_list = backlinks[backlinks['url'].isin(second_hop_links)]['back_url'].tolist()
	third_hop_links += backlink_list
	if len(third_hop_links) == 0:
		return 0
	r = get_reward(url, A_company, company_urls)
	if r > 0:
		return gamma**3
	return 0

##-----------------------------------------------------------
##-------- DQN Agent ----------------------------------------
class CrawlerAgent(object):
	def __init__(self, weights_shape, train_save_location, tf_model_folder, gamma=0.99, learning_rate=0.01):
		# Set up training parameters and TF placeholders
		self.gamma = gamma  # discount factor
		self.learning_rate = learning_rate
		self.weights_shape = weights_shape
		self.state = tf.placeholder(dtype=tf.float32, shape=[None, self.weights_shape])
		self.next_state = tf.placeholder(dtype=tf.float32, shape=[None, self.weights_shape])
		self.reward = tf.placeholder(dtype=tf.float32)
		self.is_terminal = tf.placeholder(dtype=tf.float32)
		self.build_target_net()

		# Train/test results dicts + model saving
		self.train_results_dict = OrderedDict([('pages_crawled', []), ('total_reward', []), ('terminal_states', []), ('nn_loss', [])])
		self.test_results_dict = OrderedDict([('pages_crawled', []), ('total_reward', []), ('terminal_states', [])])
		self.train_save_location = train_save_location
		# self.test_save_location = test_save_location
		self.tf_model_folder = tf_model_folder

	def build_target_net(self):
		self.weights = tf.get_variable("weights", [self.weights_shape, 1], 
			initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001))
		self.bias = tf.get_variable('bias', [1], initializer=tf.constant_initializer(0.001))
		self.v = tf.matmul(self.state, self.weights) + self.bias
		self.v_next = tf.matmul(self.next_state, self.weights) + self.bias
		self.target = self.reward + (1-self.is_terminal) * self.gamma * tf.stop_gradient(tf.reduce_max(self.v_next))
		self.loss = tf.square(self.target - self.v)/2
		self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
		
	def save_train_results(self):
		train_results_df = pd.DataFrame(self.train_results_dict)
		train_results_df.to_csv(self.train_save_location, index=False, header=True)

	# def save_test_results(self):
	# 	test_results_df = pd.DataFrame(self.test_results_dict)
	# 	test_results_df.to_csv(self.test_save_location, index=False, header=True)

	def save_tf_model(self, tf_session, tf_saver):
		tf_saver.save(tf_session, "".join([self.tf_model_folder, "tf_model"]))


##-----------------------------------------------------------
##-----------------------------------------------------------
def main():
	##-------------------- Parameters
	cycle_freq = 50
	term_steps = 50
	num_steps = 50000  # no. crawled pages before stopping
	print_freq = 1000
	start_eps = 0.2
	end_eps = 0.05
	eps_decay = 2 / num_steps
	epsilon = start_eps
	gamma = 0.9
	learning_rate = 0.001
	reload_model = False

	##-------------------- Read in data
	# Read in all URls, backlinks data and list of keywords
	links_df = pd.read_csv('data/links_dataframe.csv')
	url_list = links_df['url'].tolist()
	url_list = [l.replace("http://", "").replace("https://", "") for l in url_list if type(l) is str if l[-4:] not in [".png", ".jpg", ".pdf", ".txt"]]
	url_set = set(url_list)
	backlinks = pd.read_csv('data/backlinks_clean.csv')
	words_list = pd.read_csv("data/segmented_words_df.csv")['word'].tolist()
	word_dict = dict(zip(words_list, list(range(len(words_list)))))
	count_vec = CountVectorizer(vocabulary=word_dict)
	weights_shape = len(words_list)

	# Read in company URLs
	reward_urls = [l.replace("www.", "") for l in links_df[links_df['hops']==0]['url'].tolist()]
	reward_urls = list(set(reward_urls))
	A_company = init_automaton(reward_urls)  # Aho-corasick automaton for companies
	A_company.make_automaton()

	# Set paths
	if args.run == "no-revisit":
		revisit = False
		weights_shape += 1
		all_urls_file = RESULTS_FOLDER + "all_urls.csv"
		model_save_file = MODEL_FOLDER + "tree_model"
	else:
		revisit = True
		all_urls_file = RESULTS_FOLDER + "all_urls_revisit.csv"
		model_save_file = MODEL_FOLDER + "tree_model_revisit"

	##------------------- Initialize Crawler Agent and TF graph/session
	step_count = 0; pages_crawled = 0; total_reward = 0; terminal_states = 0
	recent_urls = []; reward_pages = []; found_rewards = []; reward_domain_set = set()

	# tf.reset_default_graph()
	# agent = CrawlerAgent(weights_shape, results_save_file, model_save_file, gamma=gamma, learning_rate=learning_rate)
	# init = tf.global_variables_initializer()
	# saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(init)

		if reload_model == True:
			print("Reloading model...")
			saver = tf.train.import_meta_graph(model_save_file+"/tf_model.meta")
			saver.restore(sess, tf.train.latest_checkpoint(model_save_file))
			all_vars = tf.get_collection('vars')

		else:
			##------------------ Run and train crawler agent -----------------------
			print("Training DQN agent...")
			if os.path.isfile(all_urls_file):
				os.remove(all_urls_file)

			while step_count < num_steps:
				url = random.choice(list(url_set - set(recent_urls)))  # don't start at recent URL








			##-------------------------------------------------------------------------

			# agent.save_tf_model(sess, saver)

	sess.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--run', default='')
	args = parser.parse_args()
	main()