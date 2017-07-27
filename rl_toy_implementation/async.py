import os
import sys
import re
import csv
import time
import random
import pickle
import argparse
import threading
import ahocorasick
import multiprocessing
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

RESULTS_FOLDER = "results/async_results/"
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
	except (UnicodeError, ValueError):
		return []
	try:
		link_list = [l.url.replace("http://", "").replace("https://", "") for l in page.links if l.url[:4] == "http"]
		link_list = link_list + [l.replace("www.", "") for l in link_list]
	except UnicodeDecodeError:
		return []
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

def pad_shorten(s, max_len):
	"""Pad URLs to max length"""
	try:
		s_out = np.pad(s, (0, max_len-len(s)), 'constant', constant_values=(0,0))
	except ValueError as e:
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

def epsilon_greedy(epsilon, action_list):
	"""Returns index of chosen action"""
	if random.uniform(0, 1) > epsilon:
		return np.argmax(action_list)
	else:
		return random.randint(0, len(action_list)-1)

def update_target_graph(from_scope,to_scope):
	from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
	to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

	op_holder = []
	for from_var,to_var in zip(from_vars,to_vars):
		op_holder.append(to_var.assign(from_var))
	return op_holder

##-----------------------------------------------------------
##-------- Q Network ----------------------------------------
class QNetwork(object):
	def __init__(self, filter_sizes, num_filters, embeddings, embedding_size, max_len, 
		gamma, scope, trainer):

		with tf.variable_scope(scope):
			self.filter_sizes = filter_sizes
			self.num_filters = num_filters
			self.embeddings = embeddings
			self.embedding_size = embedding_size
			self.max_len = max_len
			self.num_filters_total = self.num_filters * len(self.filter_sizes)

			# Set up training parameters and TF placeholders
			self.gamma = gamma  # discount factor
			self.state = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len])
			self.next_state = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len])
			self.reward = tf.placeholder(dtype=tf.float32)
			self.is_terminal = tf.placeholder(dtype=tf.float32)

			#---- Do for value function of current state
			embedded_state = tf.nn.embedding_lookup(self.embeddings, self.state)
			embedded_state = tf.expand_dims(embedded_state, -1)
			embedded_state = tf.nn.l2_normalize(embedded_state, 1)

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
					conv = tf.nn.conv2d(tf.cast(embedded_state, tf.float32), W, 
						strides=[1, 1, 1, 1], padding="VALID", name="conv")
					h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
					pooled = tf.nn.max_pool(h, ksize=[1, self.max_len-filter_size+1, 1, 1],
						strides=[1, 1, 1, 1], padding='VALID', name="pool")
					pooled_outputs.append(pooled)

			# Fully connected for s
			h_pool = tf.concat(pooled_outputs, 3)
			h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])
			self.v = tf.nn.sigmoid(tf.matmul(h_pool_flat, W_out) + b_out)


			#---- Do for value function of next state
			embedded_next_state = tf.nn.embedding_lookup(self.embeddings, self.next_state)
			embedded_next_state = tf.expand_dims(embedded_next_state, -1)
			embedded_next_state = tf.nn.l2_normalize(embedded_next_state, 1)

			W_out_target = tf.get_variable("W_out_target", [self.num_filters_total, 1],
				initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001))
			b_out_target = tf.get_variable("b_out_target", [1], initializer=tf.constant_initializer(0.001))

			# Convolutions for s'
			pooled_outputs_next = []
			for i, filter_size in enumerate(self.filter_sizes):
				with tf.name_scope("target-conv-maxpool-%s" % filter_size):
					filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
					W_target = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W_target")
					b_target = tf.Variable(tf.constant(0.01, shape=[self.num_filters]), name="b_target")
					conv_next = tf.nn.conv2d(tf.cast(embedded_next_state, tf.float32), W_target, 
						strides=[1, 1, 1, 1], padding="VALID", name="conv")
					h_next = tf.nn.relu(tf.nn.bias_add(conv_next, b_target), name="relu")
					pooled_next = tf.nn.max_pool(h_next, ksize=[1, self.max_len-filter_size+1, 1, 1],
						strides=[1, 1, 1, 1], padding='VALID', name="pool")
					pooled_outputs_next.append(pooled_next)

			# Fully connected for s'
			h_pool_next = tf.concat(pooled_outputs_next, 3)
			h_pool_flat_next = tf.reshape(h_pool_next, [-1, self.num_filters_total])
			self.v_next = tf.nn.sigmoid(tf.matmul(h_pool_flat_next, W_out_target) + b_out_target)


			#---- Loss function and gradients
			if scope != 'global':
				self.max_v_next = tf.reshape(tf.stop_gradient(tf.reduce_max(self.v_next)), [-1, 1])
				self.target = self.reward + (1-self.is_terminal) * self.gamma * self.max_v_next
				self.loss = tf.square(self.target - self.v) / 2
				# self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

				# Get gradients from local network using local losses
				local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
				self.gradients = tf.gradients(self.loss,local_vars)

				# Apply local gradients to global network
				global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
				self.apply_grads = trainer.apply_gradients(zip(self.gradients, global_vars))

##-----------------------------------------------------------
##-------- DQN Agent ----------------------------------------
class Worker():
	def __init__(self, name, cycle_freq, term_steps, num_steps, start_eps, 
		end_eps, eps_decay, gamma, max_len, embeddings, embedding_size, filter_sizes,
		num_filters, url_list, count_vec, copy_steps, trainer, A_company, reward_urls):
		self.cycle_freq = cycle_freq
		self.term_steps = term_steps
		self.num_steps = num_steps
		self.start_eps = start_eps
		self.end_eps = end_eps
		self.eps_decay = eps_decay
		self.epsilon = self.start_eps
		self.gamma = gamma
		self.max_len = max_len
		self.embedding_size = embedding_size
		self.embeddings = embeddings
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters
		self.url_list = url_list
		self.url_set = set(url_list)
		self.count_vec = count_vec
		self.copy_steps = copy_steps
		self.A_company = A_company
		self.reward_urls = reward_urls
		self.trainer = trainer

		self.name = "worker_" + str(name)
		self.local_Q = QNetwork(self.filter_sizes, self.num_filters, self.embeddings, 
			self.embedding_size, self.max_len, self.gamma,
			self.name, self.trainer)
		self.update_local_ops = update_target_graph('global',self.name)

		self.step_count = 0; self.pages_crawled = 0; self.total_reward = 0; 
		self.terminal_states = 0; self.recent_urls = []

	def train(self, sess, saver, coord):
		"""Run agent in environment and train and update parameters"""
		print ("Starting worker " + str(self.name))
		with sess.as_default(), sess.graph.as_default():
			while not coord.should_stop():

				while self.step_count < self.num_steps:
					url = get_random_url(self.url_list, self.recent_urls)
					self.steps_without_terminating = 0

					while self.step_count < self.num_steps:
						self.step_count += 1

						# Keep track of recent URLs (to avoid loops)
						self.recent_urls.append(url)
						if len(self.recent_urls) > self.cycle_freq:
							self.recent_urls = self.recent_urls[-self.cycle_freq:]

						# Get rewards and remove domain if reward
						r, reward_url_idx = get_reward(url, self.A_company, self.reward_urls)
						self.pages_crawled += 1
						self.total_reward += r
						
						# Feature representation of current page (state) and links in page
						state = build_url_feature_matrix(self.count_vec, [url], self.embeddings, self.max_len)
						link_list = get_list_of_links(url)
						link_list = set(link_list).intersection(self.url_set)
						link_list = list(link_list - set(self.recent_urls))

						# Check if terminal state
						if r > 0 or len(link_list) == 0:
							self.terminal_states += 1
							is_terminal = 1
							next_state_array = np.zeros(shape=(1, self.max_len))  # doesn't matter what this is
						else:
							is_terminal = 0
							self.steps_without_terminating += 1
							next_state_array = build_url_feature_matrix(self.count_vec, link_list,
								self.embeddings, self.max_len)

						# Train DQN
						train_dict = {
								self.local_Q.state: state, self.local_Q.next_state: next_state_array, 
								self.local_Q.reward: r, self.local_Q.is_terminal: is_terminal
						}
						loss, v_next  = sess.run([self.local_Q.loss, self.local_Q.v_next], feed_dict=train_dict)
						v_next = v_next.reshape(-1)

						# Decay epsilon
						if self.epsilon > self.end_eps:
							self.epsilon = self.epsilon - self.eps_decay

						# Choose next URL (and check for looping)
						if is_terminal == 1:
							break
						if self.steps_without_terminating >= self.term_steps:  # to prevent cycles
							break
						a = epsilon_greedy(self.epsilon, v_next)
						url = link_list[a]

						# Copy parameters to local network
						if self.pages_crawled % self.copy_steps == 0:
							sess.run(self.update_local_ops)
							self.update_progress()


	def update_progress(self):
		print("\nWorker {}: Crawled {} pages, total reward = {}, # terminal states = {}"\
		.format(self.name, self.pages_crawled, self.total_reward, self.terminal_states))


def main():
	##-------------------- Parameters
	cycle_freq = 50
	term_steps = 15
	copy_steps = 100
	num_steps = 2000  # no. crawled pages before stopping
	print_freq = 1000
	start_eps = 0.1
	end_eps = 0
	eps_decay = 2.5 / num_steps
	epsilon = start_eps
	gamma = 0.75
	learning_rate = 0.001
	reload_model = True

	max_len = 50
	embedding_size = 300
	filter_sizes = [1, 2, 3, 4]
	num_filters = 4

	##-------------------- Read in data
	print("Loading data...")
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
	embeddings = np.loadtxt('../../url_embeddings_matrix.csv', delimiter=',')

	# URL words
	words_list = pd.read_csv("../../embedded_url_word_list.csv")['word'].tolist()
	word_dict = dict(zip(words_list, list(range(len(words_list)))))
	count_vec = CountVectorizer(vocabulary=word_dict)

	# Model path
	model_save_file = MODEL_FOLDER + "async"


	#---------------------- Initialize workers and TF graph
	tf.reset_default_graph()

	with tf.device("/cpu:0"):
		trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		master_net = QNetwork(filter_sizes, num_filters, embeddings, embedding_size,
			max_len, gamma, 'global', None)

		# num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
		num_workers = 2
		workers = []
		# Create worker classes
		for i in range(num_workers):
			workers.append(Worker(i, cycle_freq, term_steps, num_steps, start_eps, end_eps,
				eps_decay, gamma, max_len, embeddings, embedding_size, filter_sizes,
				num_filters, url_list, count_vec, copy_steps, trainer, A_company, reward_urls))
			saver = tf.train.Saver()


	#---------------------- Run workers
	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		sess.run(tf.global_variables_initializer())

		worker_threads = []
		for worker in workers:
			worker_work = lambda: worker.train(sess, saver, coord)
			t = threading.Thread(target=(worker_work))
			t.start()
			time.sleep(0.5)
			worker_threads.append(t)
		coord.join(worker_threads)



if __name__ == "__main__":
	main()





