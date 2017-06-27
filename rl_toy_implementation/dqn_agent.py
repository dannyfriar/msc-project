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

def build_url_feature_vector(A_company, search_list, string_to_search):
	"""Presence of search_list words in string, along with length of string"""
	feature_vector = check_strings(A_company, search_list, string_to_search)
	feature_vector.append(len(string_to_search))
	# if sum(check_strings(A_found, found_list, string_to_search)) > 0:
	# 	feature_vector.append(1)
	# else:
	# 	feature_vector.append(0)
	return feature_vector

def build_url_feature_matrix(word_dict, url_list):
	"""Return 2d numpy array of booleans"""
	count_vec = CountVectorizer(vocabulary=word_dict)
	return count_vec.transform(url_list).toarray()


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
##-------- DQN Agent ----------------------------------------
class CrawlerAgent(object):
	def __init__(self, words_list, gamma=0.99, learning_rate=0.01,
		train_save_location="results/linear_dqn_results/dqn_crawler_train_results_revisit.csv",
		test_save_location="results/linear_dqn_results/dqn_crawler_test_results.csv",
		# tf_model_folder="models/linear_model"
		tf_model_folder="models/linear_model_revisit"):

		# Set up training parameters and TF placeholders
		self.gamma = gamma  # discount factor
		self.learning_rate = learning_rate
		self.words = words_list

		self.state = tf.placeholder(dtype=tf.float32, shape=[None, len(self.words)])
		self.next_state = tf.placeholder(dtype=tf.float32, shape=[None, len(self.words)])
		self.reward = tf.placeholder(dtype=tf.float32)
		self.is_terminal = tf.placeholder(dtype=tf.float32)
		self.build_target_net()

		# Train/test results dicts + model saving
		self.train_results_dict = OrderedDict([('pages_crawled', []), ('total_reward', []), ('terminal_states', []), ('nn_loss', [])])
		self.test_results_dict = OrderedDict([('pages_crawled', []), ('total_reward', []), ('terminal_states', [])])
		self.train_save_location = train_save_location
		self.test_save_location = test_save_location
		self.tf_model_folder = tf_model_folder

	def build_target_net(self):
		self.weights = tf.get_variable("weights", [len(self.words), 1], 
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

	def save_test_results(self):
		test_results_df = pd.DataFrame(self.test_results_dict)
		test_results_df.to_csv(self.test_save_location, index=False, header=True)

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
	epsilon = 0.05
	gamma = 0.90
	learning_rate = 0.001
	reload_model = False

	##-------------------- Read in data
	# # Read in all URls, backlinks data and list of keywords
	links_df = pd.read_csv('data/links_dataframe.csv')
	url_list = links_df['url'].tolist()
	url_list = [l.replace("http://", "").replace("https://", "") for l in url_list if type(l) is str if l[-4:] not in [".png", ".jpg", ".pdf", ".txt"]]
	url_set = set(url_list)
	backlinks = pd.read_csv('data/backlinks_clean.csv')
	words_list = pd.read_csv("data/segmented_words_df.csv")['word'].tolist()
	word_dict = dict(zip(words_list, list(range(len(words_list)))))
	count_vec = CountVectorizer(vocabulary=word_dict)

	# Read in company URLs
	reward_urls = [l.replace("www.", "") for l in links_df[links_df['hops']==0]['url'].tolist()]
	reward_urls = list(set(reward_urls))
	A_company = init_automaton(reward_urls)  # Aho-corasick automaton for companies
	A_company.make_automaton()

	##------------------- Initialize Crawler Agent and TF graph/session
	step_count = 0; pages_crawled = 0; total_reward = 0; terminal_states = 0
	found_rewards = ['dummyreward12334']
	recent_urls = []; reward_pages = []; reward_domain_set = set()
	A_found = init_automaton(found_rewards)
	A_found.make_automaton()

	tf.reset_default_graph()
	agent = CrawlerAgent(words_list, gamma=gamma, learning_rate=learning_rate)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(init)

		if reload_model == True:
			print("Reloading model...")
			# saver = tf.train.import_meta_graph('models/linear_model/tf_model.meta')
			# saver.restore(sess, tf.train.latest_checkpoint('models/linear_model/'))
			saver = tf.train.import_meta_graph('models/linear_model_revisit/tf_model.meta')
			saver.restore(sess, tf.train.latest_checkpoint('models/linear_model_revisit/'))
			all_vars = tf.get_collection('vars')
			weights_df = pd.DataFrame.from_dict({'words':words_list, 'coef': agent.weights.eval().reshape(-1).tolist()})
			# weights_df.to_csv("results/feature_coefficients.csv", index=False, header=True)
			weights_df.to_csv("results/linear_dqn_results/feature_coefficients_revisit.csv", index=False, header=True)

			# # Test a URL
			# test_url = "www.yellow-eyedpenguin.com"                                                                                              
			# s = np.array(build_url_feature_vector(agent.A, agent.words, test_url))
			# v = sess.run(agent.v, feed_dict={agent.state: s.reshape(1, -1)})
			# print("Value of URL {} is {}".format(test_url, float(v)))

		else:
			##------------------ Run and train crawler agent -----------------------
			print("Training DQN agent...")
			if os.path.isfile("results/linear_dqn_results/all_urls.csv"):
				os.remove("results/linear_dqn_results/all_urls.csv")

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
					agent.train_results_dict['total_reward'].append(total_reward)
					if r > 0:
						# found_rewards.append(reward_urls[reward_url_idx])
						# A_found = init_automaton(found_rewards)
						# A_found.make_automaton()
						reward_pages.append(url)
						# reward_domain_set.update(lookup_domain_name(links_df, reward_urls[reward_url_idx]))
						# reward_urls.pop(reward_url_idx)
						# A_company = init_automaton(reward_urls)  # Aho-corasick automaton for companies
						# A_company.make_automaton()
					
					# Feature representation of current page (state) and links in page
					state = build_url_feature_matrix(words_list, [url])
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
						next_state_array = build_url_feature_matrix(words_list, link_list)
					agent.train_results_dict['terminal_states'].append(terminal_states)

					# Train DQN
					train_dict = {
							agent.state: state, agent.next_state: next_state_array, 
							agent.reward: r, agent.is_terminal: is_terminal
					}
					opt, loss, v_next  = sess.run([agent.opt, agent.loss, agent.v_next], feed_dict=train_dict)
					agent.train_results_dict['nn_loss'].append(float(loss))

					# Print progress + save transitions
					progress_bar(step_count+1, num_steps)
					if step_count % print_freq == 0:
						print("\nCrawled {} pages, total reward = {}, # terminal states = {}, remaining rewards = {}"\
						.format(pages_crawled, total_reward, terminal_states, len(reward_urls)))
					agent.train_results_dict['pages_crawled'].append(pages_crawled)

					with open("results/linear_dqn_results/all_urls.csv", "a") as csv_file:
						writer = csv.writer(csv_file, delimiter=',')
						writer.writerow([url, r, is_terminal])

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
			agent.save_train_results()
			agent.save_tf_model(sess, saver)

			df = pd.DataFrame(reward_pages, columns=["rewards_pages"])
			df.to_csv('results/linear_dqn_results/dqn_reward_pages_revisit.csv', index=False)

	sess.close()


if __name__ == "__main__":
	main()