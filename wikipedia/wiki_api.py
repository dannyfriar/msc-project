import os
import re
import sys
import csv
import time
import gensim
import random
import pickle
import numpy as np
import pandas as pd
import ujson as json
import tensorflow as tf

from wordsegment import segment
from urllib.request import urlopen

from nltk.corpus import stopwords, words, names
stops = set(stopwords.words("english"))
eng_words = set(words.words())

##---------------- Misc functions
def progress_bar(value, endvalue, bar_length=20):
	percent = float(value) / endvalue
	arrow = '-' * int(round(percent * bar_length)-1) + '>'
	spaces = ' ' * (bar_length - len(arrow))
	sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
	sys.stdout.flush()

def save_results_list(filename, results_list):
	"""Save results to given file"""
	with open(filename, "a") as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		writer.writerow(results_list)

##---------------- To call Wikipedia API
class WikiCrawler(object):
	def __init__(self):
		self.url_base = ''.join(['https://en.wikipedia.org/w/api.php?action=query&titles={}',
			'&prop=links&pllimit=max&format=json'])
		self.random_url = ''.join(['https://en.wikipedia.org/w/api.php?action=query&generator',
			'=random&grnnamespace=0&prop=links&pllimit=max&format=json'])
		self.title_blacklist = ['User', 'Talk', 'Wikipedia', 'Help', 'Template', 
			'Category', 'Portal', 'File']

	def title_in_blacklist(self, title):
		"""Check if title contains words in blacklist"""
		if sum([t in title for t in self.title_blacklist]) > 0:
			return True

	def get_wiki_links(self, random, url_title):
		"""Gets wikipedia URLs given page title"""
		if random == True:
			url = self.random_url
		else:
			url = self.url_base.format(url_title)
			url = url.replace(' ', '%20')
		try:
			response = urlopen(url).read().decode('UTF-8')
		except UnicodeEncodeError as e:
			return url_title, []

		response = json.loads(response)['query']['pages']
		pages_dict = response[next(iter(response))]
		title = pages_dict['title']
		try:
			links = pages_dict['links']
			title_links_list = [d['title'] for d in links]
			title_links_list = [t for t in title_links_list if not self.title_in_blacklist(t)]
		except KeyError as e:
			title_links_list = []
		return title, title_links_list

	def choose_random_title(self, title_list):
		"""Choose random title from list not in blacklist"""
		title = 'User'
		while self.title_in_blacklist(title) == True:
			title = random.choice(title_list)
		return title

##---------------- RL functions
def get_reward(title):
	"""Give reward if philosophy page found"""
	if 'philosophy' in title.lower():
		return 1
	return 0

def build_feature_matrix(title_list, model, max_len):
	"""Build feature matrix from URL string"""
	feature_mat = []; sequence_lengths = []
	for title in title_list:
		try:
			word_list = [s.lower() for s in title.split(' ')]
			word_list = list(set(word_list).intersection(eng_words) - stops)
			if len(word_list) == 0:
				raise KeyError
			elif len(word_list) > max_len:
				word_list = word_list[:max_len]
			word_list = np.array([model[word] for word in word_list])
		except (IndexError, KeyError) as e:
			word_list = np.zeros((1, 300))
		sequence_lengths.append(word_list.shape[0])
		word_list = np.resize(word_list, (max_len, 300))
		word_list = word_list.reshape((1, max_len, 300))
		feature_mat.append(word_list)

	sequence_lengths = np.array(sequence_lengths).reshape(-1)
	feature_mat = np.concatenate(feature_mat)
	return feature_mat, sequence_lengths

def epsilon_greedy(epsilon, action_list):
	"""Returns index of chosen action"""
	if random.uniform(0, 1) > epsilon:
		return np.argmax(action_list)
	else:
		return random.randint(0, len(action_list)-1)


##-------------------------- Buffer
class Buffer(object):
	def __init__(self):
		self.min_buffer_size = 1
		self.max_buffer_size = 500
		self.buffer = []

	def update(self, s, r, s_prime, is_terminal, s_seq_len, s_prime_seq_len):
		self.buffer.append((s, s_prime, r, is_terminal, s_seq_len, s_prime_seq_len))
		if len(self.buffer) > self.max_buffer_size:
			self.buffer = self.buffer[1:]

	def sample(self, state, next_state, reward, is_terminal, 
		state_seq_len, next_state_seq_len):
		if len(self.buffer) >= self.min_buffer_size:
			idx = random.randint(0, len(self.buffer)-1)
			buffer_tuple = self.buffer[idx]

			train_dict = {
				state: buffer_tuple[0], next_state: buffer_tuple[1],
				reward: buffer_tuple[2], is_terminal: buffer_tuple[3],
				state_seq_len: buffer_tuple[4],
				next_state_seq_len: buffer_tuple[5]
			}
			return True, train_dict
		return False, None

##-------------------------- DQN Agent
class CrawlerAgent(object):
	def __init__(self, gamma, learning_rate, max_len, cell_size, model_save_path):
		self.gamma = gamma
		self.learning_rate = learning_rate
		self.max_len = max_len
		self.cell_size = cell_size
		self.model_save_path = model_save_path
		self.state = tf.placeholder(dtype=tf.float32, shape=[1, max_len, 300])
		self.next_state = tf.placeholder(dtype=tf.float32, shape=[None, max_len, 300])
		self.reward = tf.placeholder(dtype=tf.float32)
		self.is_terminal = tf.placeholder(dtype=tf.float32)
		self.state_seq_len = tf.placeholder(dtype=tf.int32, shape=[None])
		self.next_state_seq_len = tf.placeholder(dtype=tf.int32, shape=[None])
		self.lstm_target()
		self.lstm_network()

	def lstm_target(self):
		"""Target network for v'"""
		W_out_target = tf.get_variable("W_out_target", [self.cell_size, 1],
			initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001))
		b_out_target = tf.get_variable("b_out_target", [1], initializer=tf.constant_initializer(0.001))

		with tf.variable_scope('target_lstm_scope'):
			# State
			lstm = tf.contrib.rnn.BasicLSTMCell(num_units=self.cell_size)
			sp_out, sp_state = tf.nn.dynamic_rnn(cell=lstm, dtype=tf.float32, inputs=self.next_state,
				sequence_length=self.next_state_seq_len)
			self.v_next = tf.matmul(sp_out[:, -1, :], W_out_target) + b_out_target

	def lstm_network(self):
		"""Q-network to compute v"""
		W_out = tf.get_variable("W_out", [self.cell_size, 1],
			initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001))
		b_out = tf.get_variable("b_out", [1], initializer=tf.constant_initializer(0.001))

		with tf.variable_scope('lstm_scope'):
			# State
			lstm = tf.contrib.rnn.BasicLSTMCell(num_units=self.cell_size)
			s_out, s_state = tf.nn.dynamic_rnn(cell=lstm, dtype=tf.float32, inputs=self.state,
				sequence_length=self.state_seq_len)
			self.v = tf.matmul(s_out[:, -1, :], W_out) + b_out

		# Compute target and incur loss
		self.max_v_next = tf.to_float(tf.stop_gradient(tf.reduce_max(self.v_next)))
		self.target = self.reward + (1-self.is_terminal) * self.gamma * self.max_v_next
		self.v = tf.to_float(self.v)
		self.loss = tf.square(self.target - self.v) / 2
		self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

	def update_target_net(self, sess, tf_vars):
		"""Copy network weights from main net to target net"""
		num_vars = int(len(tf_vars)/2)
		op_list = []
		for idx, var in enumerate(tf_vars[num_vars:]):
			op_list.append(tf_vars[idx].assign(var.value()))

		# Run the TF operations to copy variables
		for op in op_list:
			sess.run(op)

	def save_tf_model(self, sess, tf_saver):
		tf_saver.save(sess, "/".join([self.model_save_path, "tf_model"]))


##-----------------------------
def main():
	num_crawls = 10000
	print_freq = 100
	copy_steps = 50
	cycle_freq = 10
	term_steps = 15
	gamma = 0.9
	epsilon = 0.05
	learning_rate = 0.001
	max_len = 10
	cell_size = 32
	reload_model = False

	print("#------- Loading data...")
	model = gensim.models.Word2Vec.load_word2vec_format('../../GoogleNews-vectors-negative300.bin', 
		binary=True, limit=100000)
	with open('first_hops.pkl', 'rb') as f:
		first_hop_start = pickle.load(f)
	with open('second_hops.pkl', 'rb') as f:
		second_hop_start = pickle.load(f)
	# start_pages = first_hop_start + second_hop_start
	start_pages = first_hop_start

	results_file = 'wiki_crawl_results.csv'
	if os.path.isfile(results_file):
		os.remove(results_file)

	#---------------- Starting crawler
	print("#------- Starting crawler...")
	steps = 0; rewards = 0; terminal_states = 0; recent_titles = []

	tf.reset_default_graph()
	wiki_crawl = WikiCrawler()
	replay_buffer = Buffer()
	agent = CrawlerAgent(gamma, learning_rate, max_len, cell_size, 'lstm_model')
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(init)

		if reload_model == True:
			print("Reloading model...")
			saver = tf.train.import_meta_graph("lstm_model/tf_model.meta")
			saver.restore(sess, tf.train.latest_checkpoint("lstm_model"))
			all_vars = tf.get_collection('vars')

		while steps < num_crawls:
			# title, title_links_list = wiki_crawl.get_wiki_links(True, None)
			title, title_links_list = wiki_crawl.get_wiki_links(False, random.choice(start_pages))
			steps_before_term = 0

			while steps < num_crawls:
				is_terminal = 0
				steps += 1
				steps_before_term += 1
				progress_bar(steps, num_crawls)
				state, state_seq_lens = build_feature_matrix([title], model, max_len)

				# Keep track of recent URLs (to avoid loops)
				recent_titles.append(title)
				if len(recent_titles) > cycle_freq:
					recent_titles = recent_titles[-cycle_freq:]
				title_links_list = list(set(title_links_list) - set(recent_titles))

				# Check for reward
				r = get_reward(title)
				if r > 0 or len(title_links_list) == 0:
					is_terminal = 1
					next_state = np.zeros((1, max_len, 300))
					next_state_seq_lens = np.zeros(1)
				else:
					next_state, next_state_seq_lens = build_feature_matrix(title_links_list, model, max_len)
				rewards += r
				terminal_states += is_terminal

				# Train neural network
				train_dict = {
					agent.state: state, agent.next_state: next_state,
					agent.reward: r, agent.is_terminal: is_terminal,
					agent.state_seq_len: state_seq_lens,
					agent.next_state_seq_len: next_state_seq_lens,
				}
				opt, v_next  = sess.run([agent.opt, agent.v_next], feed_dict=train_dict)

				# Copy parameters every copy_steps transitions
				if steps % copy_steps == 0:
					agent.update_target_net(sess, tf.trainable_variables())

				# Save results and continue
				save_results_list(results_file, [title, r, is_terminal])
				if steps % print_freq == 0:
					print("\nCrawled {} pages, {} rewards, {} terminal_states".format(steps, 
						rewards, terminal_states))
					agent.save_tf_model(sess, saver)
				if is_terminal == 1 or steps_before_term >= term_steps:
					break
				
				a = epsilon_greedy(epsilon, v_next)
				# next_title = choose_random_title(title_links_list)
				next_title = title_links_list[a]
				title, title_links_list = wiki_crawl.get_wiki_links(False, next_title)

		sess.close()

	print("\nCrawled {} pages, {} rewards, {} terminal_states".format(steps, rewards, terminal_states))

if __name__ == "__main__":
	main()

