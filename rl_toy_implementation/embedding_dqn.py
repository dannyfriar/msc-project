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
			return [], [], ''
	except (UnicodeError, ValueError) as e:
		return [], [], ''
	try:
		link_list = [l.url.replace("http://", "").replace("https://", "") for l in page.links if l.url[:4] == "http"]
		link_list = link_list + [l.replace("www.", "") for l in link_list]
		text_list = [l.text.lower() for l in page.links if l.url[:4] == "http"]
		text_list = text_list + text_list
		page_title = page.title.lower()
	except UnicodeDecodeError:
		return [], [], ''
	return link_list, text_list, page_title

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

def build_url_feature_matrix(count_vec, text_count_vec, url_list, text_list, 
	embeddings, embeddings_anchor, max_len):
	"""Return 2d numpy array of booleans"""
	feature_matrix = count_vec.transform(url_list).toarray()
	text_feature_matrix = text_count_vec.transform(text_list).toarray()
	if len(url_list) == 1:
		s = np.nonzero(feature_matrix)[1]
		s = np.pad(s, (0, max_len-len(s)), 'constant', constant_values=(0,0)).reshape(1, -1)
	else:
		s = np.nonzero(feature_matrix)
		splits = np.cumsum(np.bincount(s[0]))
		s = np.split(s[1], splits.tolist())[:-1]
		s = [np.pad(w, (0, max_len-len(w)), 'constant', constant_values=(0,0)) for w in s]
		s = np.stack(s, axis=0)

	if len(text_list) == 1:
		s_text = np.nonzero(text_feature_matrix)[1]
		s_text = np.pad(s_text, (0, max_len-len(s_text)), 'constant', constant_values=(0,0)).reshape(1, -1)
	else:
		s_text = np.nonzero(text_feature_matrix)
		text_splits = np.cumsum(np.bincount(s_text[0]))
		s_text = np.split(s_text[1], text_splits.tolist())[:-1]
		s_text = [np.pad(w, (0, max_len-len(w)), 'constant', constant_values=(0,0)) for w in s_text]
		s_text = np.stack(s_text, axis=0)

	return s, s_text

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
##-------- DQN Agent ----------------------------------------
class CrawlerAgent(object):
	def __init__(self, filter_sizes, num_filters, embeddings, embedding_size, max_len,
	 tf_model_folder, gamma=0.99, learning_rate=0.01):
		# Embedding parameters
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters
		self.embeddings = embeddings
		self.embedding_size = embedding_size
		self.max_len = max_len
		self.seq_len = 2 * max_len
		self.num_filters_total = self.num_filters * len(self.filter_sizes)

		# Set up training parameters and TF placeholders
		self.gamma = gamma  # discount factor
		self.learning_rate = learning_rate
		self.state = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len])
		self.state_text = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len])
		self.next_state = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len])
		self.next_state_text = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len])
		self.reward = tf.placeholder(dtype=tf.float32)
		self.is_terminal = tf.placeholder(dtype=tf.float32)
		self.target_net()
		self.embeddings_net()
		self.tf_model_folder = tf_model_folder

	def target_net(self):
		# s1, s2 = tf.split(self.next_state, num_or_size_splits=2, axis=2)
		embedded_next_state1 = tf.nn.embedding_lookup(self.embeddings, self.next_state)
		embedded_next_state1 = tf.expand_dims(embedded_next_state1, -1)
		embedded_next_state2 = tf.nn.embedding_lookup(self.embeddings, self.next_state_text)
		embedded_next_state2 = tf.expand_dims(embedded_next_state2, -1)
		embedded_next_state = tf.concat([embedded_next_state1, embedded_next_state2], axis=3)

		W_out_target = tf.get_variable("W_out_target", [self.num_filters_total, 1],
			initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001))
		b_out_target = tf.get_variable("b_out_target", [1], initializer=tf.constant_initializer(0.001))

		# Convolutions for s'
		pooled_outputs_next = []
		for i, filter_size in enumerate(self.filter_sizes):
			with tf.name_scope("target-conv-maxpool-%s" % filter_size):
				filter_shape = [filter_size, self.embedding_size, 2, self.num_filters]
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

	def embeddings_net(self):
		# s1, s2 = tf.split(self.state, num_or_size_splits=2, axis=2)
		embedded_state1 = tf.nn.embedding_lookup(self.embeddings, self.state)
		embedded_state1 = tf.expand_dims(embedded_state1, -1)
		embedded_state2 = tf.nn.embedding_lookup(self.embeddings, self.state_text)
		embedded_state2 = tf.expand_dims(embedded_state2, -1)
		embedded_state = tf.concat([embedded_state1, embedded_state2], axis=3)

		W_out = tf.get_variable("W_out", [self.num_filters_total, 1], 
			initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001))
		b_out = tf.get_variable("b_out", [1], initializer=tf.constant_initializer(0.001))

		# Convolutions for s
		pooled_outputs = []
		for i, filter_size in enumerate(self.filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				filter_shape = [filter_size, self.embedding_size, 2, self.num_filters]
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

		# Compute target and incur loss
		max_v_next = tf.reshape(tf.stop_gradient(tf.reduce_max(self.v_next)), [-1, 1])
		target = self.reward + (1-self.is_terminal) * self.gamma * max_v_next
		self.loss = tf.square(target - self.v) / 2
		self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

	def update_target_net(self, sess, tf_vars):
		"""Copy network weights from main net to target net"""
		num_vars = int(len(tf_vars)/2)
		op_list = []
		for idx, var in enumerate(tf_vars[num_vars:]):
			op_list.append(tf_vars[idx].assign(var.value()))

		# Run the TF operations to copy variables
		for op in op_list:
			sess.run(op)

	def save_tf_model(self, tf_session, tf_saver):
		tf_saver.save(tf_session, "/".join([self.tf_model_folder, "tf_model"]))


##-----------------------------------------------------------
##-----------------------------------------------------------
def main():
	##-------------------- Parameters
	cycle_freq = 50
	term_steps = 40
	copy_steps = 100
	num_steps = 100000  # no. crawled pages before stopping
	print_freq = 1000
	start_eps = 0.1
	end_eps = 0
	eps_decay = 1.5 / num_steps
	epsilon = start_eps
	gamma = 0.75
	learning_rate = 0.001
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
	embeddings_anchor = np.loadtxt('../../anchor_embeddings_matrix.csv', delimiter=',')

	# URL words
	words_list = pd.read_csv("../../embedded_url_word_list.csv")['word'].tolist()
	word_dict = dict(zip(words_list, list(range(len(words_list)))))
	count_vec = CountVectorizer(vocabulary=word_dict)

	# Anchor text/title words
	text_words_list = pd.read_csv("../../embedded_anchor_list.csv")['word'].tolist()
	text_word_dict = dict(zip(words_list, list(range(len(text_words_list)))))
	text_count_vec = CountVectorizer(vocabulary=text_word_dict)

	# File locations
	all_urls_file = RESULTS_FOLDER + "all_urls_revisit.csv"
	model_save_file = MODEL_FOLDER + "embeddings"

	##------------------- Initialize Crawler Agent and TF graph/session
	step_count = 0; pages_crawled = 0; total_reward = 0; terminal_states = 0
	recent_urls = []; reward_pages = []; found_rewards = []; reward_domain_set = set()

	tf.reset_default_graph()
	agent = CrawlerAgent(filter_sizes, num_filters, embeddings, embedding_size, max_len,
		model_save_file, gamma=gamma, learning_rate=learning_rate)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(init)
		##------------------ Run and train crawler agent -----------------------
		print("Training DQN agent...")
		# if os.path.isfile(all_urls_file):
		# 	os.remove(all_urls_file)

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
				link_list, text_list, page_title = get_list_of_links(url)
				state, state_text = build_url_feature_matrix(count_vec, text_count_vec, [url], [page_title], 
					embeddings, embeddings_anchor, max_len)
				if len(link_list) > 0:
					url_df = pd.DataFrame.from_dict({"url":link_list, "text": text_list})
					url_df = url_df.sort_values(by="url").reset_index()
					link_list = set(link_list).intersection(url_set)
					link_list = list(link_list - set(recent_urls))
					filtered_url_df = pd.DataFrame.from_dict({"url":link_list})
					url_df = pd.merge(url_df, filtered_url_df, on='url')
					url_df = url_df.drop_duplicates(['url'])
					text_list = url_df['text'].tolist()
					text_list = [t+" www" for t in text_list]
					link_list = url_df['url'].tolist()

				# Check if terminal state
				if r > 0 or len(link_list) == 0:
					terminal_states += 1
					is_terminal = 1
					next_state = np.zeros(shape=(1, max_len))  # doesn't matter what this is
					next_state_text = np.zeros(shape=(1, max_len))  # doesn't matter what this is
				else:
					is_terminal = 0
					steps_without_terminating += 1
					next_state, next_state_text = build_url_feature_matrix(count_vec, text_count_vec,
					 link_list, text_list, embeddings, embeddings_anchor, max_len)

				# Train DQN
				train_dict = {
						agent.state: state, agent.state_text: state_text,
						agent.next_state: next_state, agent.next_state_text: next_state_text,
						agent.reward: r, agent.is_terminal: is_terminal
				}
				opt, loss, v_next  = sess.run([agent.opt, agent.loss, agent.v_next], feed_dict=train_dict)
				v_next = v_next.reshape(-1)

				# Copy parameters every copy_steps transitions
				if step_count % copy_steps == 0:
					agent.update_target_net(sess, tf.trainable_variables())

				# input("Press enter to continue...")

				# Print progress + save transitions
				progress_bar(step_count+1, num_steps)
				if step_count % print_freq == 0:
					print("\nCrawled {} pages, total reward = {}, # terminal states = {}, remaining rewards = {}"\
					.format(pages_crawled, total_reward, terminal_states, len(reward_urls)))

				# with open(all_urls_file, "a") as csv_file:
				# 	writer = csv.writer(csv_file, delimiter=',')
				# 	writer.writerow([url, r, is_terminal, float(loss)])

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
		# agent.save_tf_model(sess, saver)

	sess.close()


if __name__ == "__main__":
	random.seed(123)
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--run', default='')
	args = parser.parse_args()
	main()