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
storage = StorageEngine("/nvme/rl_project_webcache/")

from nltk.corpus import stopwords, words, names
stops = stopwords.words("english")

RESULTS_FOLDER = "results/embedding_results/"
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
		link_list = link_list + [l.replace("www.", "") for l in link_list]
	except UnicodeDecodeError:
		return []
	return link_list

def lookup_domain_name(links_df, domain_url):
	"""Returns list of all URLs within domain web site (in database)"""
	return links_df[links_df['domain'].values == domain_url]['url'].tolist()

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

def build_url_feature_matrix(count_vec, url_list, embeddings, max_len):
	"""Return 2d numpy array of booleans"""
	feature_matrix = count_vec.transform(url_list).toarray()
	if len(url_list) == 1:
		s = np.nonzero(feature_matrix)[1]
		s = np.pad(s, (0, max_len-len(s)), 'constant', constant_values=(0,0)).reshape(1, -1)
		return s
	else:
		s_prime = np.nonzero(feature_matrix)
		splits = np.cumsum(np.bincount(s_prime[0]))
		s_prime = np.split(s_prime[1], splits.tolist())[:-1]
		s_prime = [np.pad(s, (0, max_len-len(s)), 'constant', constant_values=(0,0)) for s in s_prime]
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

##-----------------------------------------------------------
##-------- Buffer -------------------------------------------
class Buffer(object):
	def __init__(self, min_buffer_size, max_buffer_size):
		self.min_buffer_size = min_buffer_size
		self.batch_size = int(min_buffer_size/2)
		self.max_buffer_size = max_buffer_size
		self.alpha = 0.5
		self.beta = 1
		self.buffer = []

	def update(self, s, r, s_prime, is_terminal):
		self.buffer.append((s, s_prime, r, is_terminal, 0))
		if len(self.buffer) > self.max_buffer_size:
			self.buffer = self.buffer[1:]

	def add_loss(self, idx, loss):
		"""Add neural net loss to buffer element"""
		b = self.buffer[idx]
		self.buffer[idx] = (b[0], b[1], b[2], b[3], loss)
		self.buffer = sorted(self.buffer, key=lambda x: int(x[4]), reverse=True)

	def sample(self, state, next_state, reward, is_terminal, sample_weight, priority):
		if len(self.buffer) >= self.min_buffer_size:
			if priority == True:
				probs = [i for i in range(1, len(self.buffer)+1)]
				probs = [p**self.alpha for p in probs]
				probs = [p/sum(probs) for p in probs]
				idx = np.abs(np.array(probs) - random.uniform(0, 1)).argmin()
				max_weight = (1/min(probs))**self.beta
				weight = (1/probs[idx])**self.beta / max_weight
			else:
				indices = random.sample(range(0, len(self.buffer)-1), self.batch_size)
				weight = 1

			buffer_sample = [self.buffer[i] for i in indices]
			s_sample = np.concatenate([b[0] for b in buffer_sample], axis=0)
			s_prime_sample = [b[1] for b in buffer_sample]
			s_prime_lengths = [s.shape[0] for s in s_prime_sample]
			s_prime_sample = np.concatenate(s_prime_sample, axis=0)
			r_sample = np.concatenate([np.array([b[2]]).reshape(-1, 1) for b in buffer_sample], axis=0)
			t_sample = np.concatenate([np.array([1-b[3]]).reshape(-1, 1) for b in buffer_sample], axis=0)

			train_dict = {
					state: s_sample, next_state: s_prime_sample,
					reward: r_sample, is_terminal: t_sample,
					sample_weight: weight
			}
			return True, -1, train_dict, s_prime_lengths
		return False, -1, None, []


##-----------------------------------------------------------
##-------- DQN Agent ----------------------------------------
class CrawlerAgent(object):
	def __init__(self, weights_shape, filter_sizes, num_filters, embeddings, embedding_size, max_len,
	 tf_model_folder, priority, min_buffer_size, max_buffer_size,
	 gamma=0.99, learning_rate=0.01):
		# Embedding parameters
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters
		self.embeddings = embeddings
		self.embedding_size = embedding_size
		self.max_len = max_len
		self.num_filters_total = self.num_filters * len(self.filter_sizes) * 41

		# Set up training parameters and TF placeholders
		self.gamma = gamma  # discount factor
		self.learning_rate = learning_rate
		self.weights_shape = weights_shape
		self.state = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len])
		self.next_state = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len])
		self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1])
		self.is_terminal = tf.placeholder(dtype=tf.float32, shape=[None, 1])
		self.sample_weight = tf.placeholder(dtype=tf.float32)
		self.v_next_splits = [1]
		self.embeddings_net()
		self.buffer = Buffer(min_buffer_size, max_buffer_size)
		self.priority = priority

		# Train/test results dicts + model saving
		self.train_results_dict = OrderedDict([('pages_crawled', []), ('total_reward', []), ('terminal_states', []), ('nn_loss', [])])
		self.test_results_dict = OrderedDict([('pages_crawled', []), ('total_reward', []), ('terminal_states', [])])
		self.tf_model_folder = tf_model_folder

	def embeddings_net(self):
		embedded_state = tf.nn.embedding_lookup(self.embeddings, self.state)
		embedded_state = tf.expand_dims(embedded_state, -1)
		embedded_next_state = tf.nn.embedding_lookup(self.embeddings, self.next_state)
		embedded_next_state = tf.expand_dims(embedded_next_state, -1)

		W_out = tf.get_variable("W_out", [self.num_filters_total, 1], 
			initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001))
		b_out = tf.get_variable("b_out", [1], initializer=tf.constant_initializer(0.001))

		# Apply convolutions
		pooled_outputs = []
		pooled_outputs_next = []
		for i, filter_size in enumerate(self.filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				# Run for s (state)		
				filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
				b = tf.Variable(tf.constant(0.01, shape=[self.num_filters]), name="b")
				conv = tf.nn.conv2d(tf.cast(embedded_state, tf.float32), W, 
					strides=[1, 1, 1, 1], padding="VALID", name="conv")
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				pooled = tf.nn.max_pool(h, ksize=[1, 10-filter_size+1, 1, 1],
					strides=[1, 1, 1, 1], padding='VALID', name="pool")
				pooled_outputs.append(pooled)

				# Run for s' (next state)
				conv_next = tf.nn.conv2d(tf.cast(embedded_next_state, tf.float32), W, 
					strides=[1, 1, 1, 1], padding="VALID", name="conv")
				h_next = tf.nn.relu(tf.nn.bias_add(conv_next, b), name="relu")
				pooled_next = tf.nn.max_pool(h_next, ksize=[1, 10-filter_size+1, 1, 1],
					strides=[1, 1, 1, 1], padding='VALID', name="pool")
				pooled_outputs_next.append(pooled_next)

		# Fully connected for s
		h_pool = tf.concat(pooled_outputs, 3)
		h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])
		self.v = tf.nn.sigmoid(tf.matmul(h_pool_flat, W_out) + b_out)

		# Fully connected for s'
		h_pool_next = tf.concat(pooled_outputs_next, 3)
		h_pool_flat_next = tf.reshape(h_pool_next, [-1, self.num_filters_total])
		self.v_next = tf.nn.sigmoid(tf.matmul(h_pool_flat_next, W_out) + b_out)

		v_next_list = tf.split(self.v_next, self.v_next_splits, 0)
		v_next_list = [tf.reduce_max(v, axis=0) for v in v_next_list]
		max_v_next = tf.reshape(tf.concat(v_next_list, axis=0), [-1, 1])

		# Compute target and incur loss
		# max_v_next = tf.reshape(tf.stop_gradient(tf.reduce_max(self.v_next)), [-1, 1])
		target = self.reward + (1-self.is_terminal) * self.gamma * max_v_next
		self.loss = tf.square(target - self.v) / 2
		self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

	def sample_buffer(self):
		train, idx, train_dict, self.v_next_splits = self.buffer.sample(self.state, 
			self.next_state, self.reward, self.is_terminal, sample_weight=self.sample_weight, 
			priority=self.priority)
		return train, idx, train_dict

	def save_tf_model(self, tf_session, tf_saver):
		tf_saver.save(tf_session, "/".join([self.tf_model_folder, "tf_model"]))


##-----------------------------------------------------------
##-----------------------------------------------------------
def main():
	##-------------------- Parameters
	cycle_freq = 50
	term_steps = 40
	num_steps = 50000  # no. crawled pages before stopping
	print_freq = 1000
	start_eps = 0.1
	end_eps = 0
	eps_decay = 2 / num_steps
	epsilon = start_eps
	gamma = 0.75
	learning_rate = 0.001
	priority = False
	train_sample_size = 100
	min_buffer_size = 2 * train_sample_size
	max_buffer_size = 2500
	reload_model = False

	max_len = 50
	embedding_size = 300
	filter_sizes = [1, 2, 3]
	num_filters = 3

	##-------------------- Read in data
	links_df = pd.read_csv("new_data/links_dataframe.csv")
	rm_list = ['aarp.org', 'akc.org', 'alcon.com', 'lincoln.com', 'orlakiely.com', 
	'red.com', 'ef.com', 'ozarksfirst.com']
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
	weights_shape = len(words_list)

	# File locations
	all_urls_file = RESULTS_FOLDER + "all_urls_revisit.csv"
	model_save_file = MODEL_FOLDER + "embeddings"

	##------------------- Initialize Crawler Agent and TF graph/session
	step_count = 0; pages_crawled = 0; total_reward = 0; terminal_states = 0
	recent_urls = []; reward_pages = []; found_rewards = []; reward_domain_set = set()

	tf.reset_default_graph()
	agent = CrawlerAgent(weights_shape, filter_sizes, num_filters, embeddings, embedding_size, max_len,
		model_save_file, priority, min_buffer_size, max_buffer_size, gamma, learning_rate)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(init)
		##------------------ Run and train crawler agent -----------------------
		print("Training DQN agent...")
		if os.path.isfile(all_urls_file):
			os.remove(all_urls_file)

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
				if r > 0:
					reward_pages.append(url)
				
				# Feature representation of current page (state) and links in page
				state = build_url_feature_matrix(count_vec, [url], embeddings, max_len)
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

				# Update buffer
				agent.buffer.update(state, r, next_state_array, is_terminal)
				train, idx, train_dict = agent.sample_buffer()

				# # Train DQN
				# train_dict = {
				# 		agent.state: state, agent.next_state: next_state_array, 
				# 		agent.reward: r, agent.is_terminal: is_terminal
				# }
				t0 = time.time()
				if train == True:
					opt, loss, v_next  = sess.run([agent.opt, agent.loss, agent.v_next], feed_dict=train_dict)
				print("Time to train = {}".format(time.time()-t0))
				t0 = time.time()
				v_next = sess.run(agent.v_next, feed_dict={agent.next_state: next_state_array}).reshape(-1)
				print("Time to get v_next = {}".format(time.time()-t0))

				if pages_crawled >= 200:
					input("Press enter to continue...")

				# Print progress + save transitions
				progress_bar(step_count+1, num_steps)
				if step_count % print_freq == 0:
					print("\nCrawled {} pages, total reward = {}, # terminal states = {}, remaining rewards = {}"\
					.format(pages_crawled, total_reward, terminal_states, len(reward_urls)))

				with open(all_urls_file, "a") as csv_file:
					writer = csv.writer(csv_file, delimiter=',')
					writer.writerow([url, r, is_terminal])

				# Decay epsilon
				if epsilon > end_eps:
					epsilon = epsilon - eps_decay

				# Choose next URL (and check for looping)
				if is_terminal == 1:
					break
				if steps_without_terminating >= term_steps:  # to prevent cycles
					break
				a = epsilon_greedy(epsilon, v_next)
				url = link_list[a]
		##-------------------------------------------------------------------------

		print("\nCrawled {} pages, total reward = {}, # terminal states = {}"\
			.format(pages_crawled, total_reward, terminal_states))
		agent.save_tf_model(sess, saver)

	sess.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--run', default='')
	args = parser.parse_args()
	main()