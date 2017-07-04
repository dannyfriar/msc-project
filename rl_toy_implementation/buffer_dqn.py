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
storage = StorageEngine("/nvme/webcache/")

from nltk.corpus import stopwords, words, names
stops = stopwords.words("english")

RESULTS_FOLDER = "results/buffer_dqn_results/"
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
	def __init__(self):
		self.min_buffer_size = 50
		self.max_buffer_size = 1000
		self.buffer = []

	def update(self, s, r, s_prime, is_terminal):
		self.buffer.append((s, s_prime, r, is_terminal))
		if len(self.buffer) > self.max_buffer_size:
			self.buffer = self.buffer[1:]

	def sample(self, state, next_state, reward, is_terminal):
		if len(self.buffer) >= self.min_buffer_size:
			idx = random.randint(0, len(self.buffer)-1)
			buffer_tuple = self.buffer[idx]
			train_dict = {
				state: buffer_tuple[0], next_state: buffer_tuple[1],
				reward: buffer_tuple[2], is_terminal: buffer_tuple[3]
			}
			return True, train_dict
		return False, None

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
		self.build_train_net()
		self.buffer = Buffer()

		# Train/test results dicts + model saving
		self.train_results_dict = OrderedDict([('pages_crawled', []), ('total_reward', []), ('terminal_states', []), ('nn_loss', [])])
		self.test_results_dict = OrderedDict([('pages_crawled', []), ('total_reward', []), ('terminal_states', [])])
		self.train_save_location = train_save_location
		self.tf_model_folder = tf_model_folder

	def build_target_net(self):
		self.t_w1 = tf.get_variable("t_w1", [self.weights_shape, 6000], initializer=tf.random_normal_initializer(mean=0.0, stddev=1/4000))
		self.t_w2 = tf.get_variable("t_w2", [6000, 300], initializer=tf.random_normal_initializer(mean=0.0, stddev=1/6000))
		self.t_w3 = tf.get_variable("t_w3", [300, 1], initializer=tf.random_normal_initializer(mean=0.0, stddev=1/300))
		self.t_b1 = tf.get_variable("t_b1", [6000], initializer=tf.constant_initializer(0.001))
		self.t_b2 = tf.get_variable("t_b2", [300], initializer=tf.constant_initializer(0.001))
		self.t_b3 = tf.get_variable("t_b3", [1], initializer=tf.constant_initializer(0.001))

		self.h1_next = tf.nn.relu(tf.matmul(self.next_state, self.t_w1) + self.t_b1)
		self.h2_next = tf.nn.relu(tf.matmul(self.h1_next, self.t_w2) + self.t_b2)
		self.v_next = tf.matmul(self.h2_next, self.t_w3) + self.t_b3
		self.target = self.reward + (1-self.is_terminal) * self.gamma * tf.stop_gradient(tf.reduce_max(self.v_next))
		
	def build_train_net(self):
		self.w1 = tf.get_variable("w1", [self.weights_shape, 6000], initializer=tf.random_normal_initializer(mean=0.0, stddev=1/4000))
		self.w2 = tf.get_variable("w2", [6000, 300], initializer=tf.random_normal_initializer(mean=0.0, stddev=1/6000))
		self.w3 = tf.get_variable("w3", [300, 1], initializer=tf.random_normal_initializer(mean=0.0, stddev=1/300))
		self.b1 = tf.get_variable("b1", [6000], initializer=tf.constant_initializer(0.001))
		self.b2 = tf.get_variable("b2", [300], initializer=tf.constant_initializer(0.001))
		self.b3 = tf.get_variable("b3", [1], initializer=tf.constant_initializer(0.001))

		self.h1 = tf.nn.relu(tf.matmul(self.state, self.w1) + self.b1)
		self.h2 = tf.nn.relu(tf.matmul(self.h1, self.w2) + self.b2)
		self.v = tf.matmul(self.h2, self.w3) + self.b3
		self.loss = tf.square(self.target - self.v)/2
		self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

	def update_target_net(self, sess, tf_vars):
		"""Copy network weights from main net to target net"""
		num_vars = int(len(tf_vars)/2)
		op_list = []
		for idx, var in enumerate(tf_vars[num_vars:]):
			op_list.append(tf_vars[idx].assign(var.value()))
		for op in op_list:
			sess.run(op)

	def sample_buffer(self):
		return self.buffer.sample(self.state, self.next_state, self.reward, self.is_terminal)

	def save_train_results(self):
		train_results_df = pd.DataFrame(self.train_results_dict)
		train_results_df.to_csv(self.train_save_location, index=False, header=True)

	def save_tf_model(self, tf_session, tf_saver):
		tf_saver.save(tf_session, "/".join([self.tf_model_folder, "tf_model"]))


##-----------------------------------------------------------
##-----------------------------------------------------------
def main():
	##-------------------- Parameters
	cycle_freq = 50
	term_steps = 50
	num_steps = 50000  # no. crawled pages before stopping
	print_freq = 1000
	copy_steps = 100
	start_eps = 0.2
	end_eps = 0.05
	eps_decay = 2 / num_steps
	epsilon = start_eps
	gamma = 0.9
	learning_rate = 0.001
	train_sample_size = 5
	reload_model = True

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
		model_save_file = MODEL_FOLDER + "deep_buffer_model"
		results_save_file = RESULTS_FOLDER + "dqn_crawler_train_results.csv"
		test_value_files = RESULTS_FOLDER + "test_value.csv"
	else:
		revisit = True
		all_urls_file = RESULTS_FOLDER + "all_urls_revisit.csv"
		model_save_file = MODEL_FOLDER + "deep_buffer_model_revisit"
		results_save_file = RESULTS_FOLDER + "dqn_crawler_train_results_revisit.csv"
		test_value_files = RESULTS_FOLDER + "test_value_revisit.csv"

	##------------------- Initialize Crawler Agent and TF graph/session
	step_count = 0; pages_crawled = 0; total_reward = 0; terminal_states = 0
	recent_urls = []; reward_pages = []; found_rewards = []; reward_domain_set = set()

	tf.reset_default_graph()
	agent = CrawlerAgent(weights_shape, results_save_file, model_save_file, gamma=gamma, learning_rate=learning_rate)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(init)

		if reload_model == True:
			print("Reloading model...")
			saver = tf.train.import_meta_graph(model_save_file+"/tf_model.meta")
			saver.restore(sess, tf.train.latest_checkpoint(model_save_file))
			all_vars = tf.get_collection('vars')

			test_urls = pd.read_csv("data/random_test_url_sample.csv")['url'].tolist()
			state_array = build_url_feature_matrix(word_dict, test_urls, revisit, found_rewards)
			v = sess.run(agent.v, feed_dict={agent.state: state_array}).reshape(-1).tolist()
			pd.DataFrame.from_dict({'url':test_urls, 'value':v}).to_csv(test_value_files, index=False)

		else:
			##------------------ Run and train crawler agent -----------------------
			print("Training DQN agent...")
			if os.path.isfile(all_urls_file):
				os.remove(all_urls_file)

			while step_count < num_steps:
				url = random.choice(list(url_set - set(recent_urls)))  # don't start at recent URL
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
						if revisit == False:
							found_rewards.append(reward_urls[reward_url_idx])
							reward_domain_set.update(lookup_domain_name(links_df, reward_urls[reward_url_idx]))
							reward_urls.pop(reward_url_idx)
							A_company = init_automaton(reward_urls)  # Aho-corasick automaton for companies
							A_company.make_automaton()
					
					# Feature representation of current page (state) and links in page
					state = build_url_feature_matrix(words_list, [url], revisit, found_rewards)
					link_list = get_list_of_links(url)
					link_list = append_backlinks(url, backlinks, link_list)
					link_list = set(link_list).intersection(url_set)
					link_list = list(link_list - set(recent_urls))

					# Check if terminal state
					if r > 0 or len(link_list) == 0:
						terminal_states += 1
						is_terminal = 1
						next_state_array = np.zeros(shape=(1, len(words_list)))  # doesn't matter what this is
					else:
						is_terminal = 0
						steps_without_terminating += 1
						next_state_array = build_url_feature_matrix(words_list, link_list, revisit, found_rewards)

					# Update buffer
					agent.buffer.update(state, r, next_state_array, is_terminal)
					train, train_dict = agent.sample_buffer()

					# Train DQN and compute values for the next states
					if train == True:
						opt, loss, v_next  = sess.run([agent.opt, agent.loss, agent.v_next], feed_dict=train_dict)
						for _ in range(train_sample_size-1):
							_, train_dict = agent.sample_buffer()
							opt, loss, _  = sess.run([agent.opt, agent.loss, agent.v_next], feed_dict=train_dict)
					v_next  = sess.run(agent.v_next, feed_dict={agent.next_state: next_state_array})

					# Copy parameters every copy_steps transitions
					if step_count % copy_steps == 0:
						agent.update_target_net(sess, tf.trainable_variables())

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