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

from evolutionai import StorageEngine
storage = StorageEngine("/nvme/webcache/")

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

def read_csv_to_list(filename):
	with open(filename) as f:  # relevant english words
		reader = csv.reader(f)
		csv_list = list(reader)
	csv_list = [c[0] for c in csv_list]
	return(csv_list)

def get_list_of_links(url, s=storage):
	"""Use the LMDB database to get a list of links for a given URL"""
	try:
		page = s.get_page(url)
	except UnicodeError:
		return []
	if page is None:
		return []
	try:
		link_list = [l.url.replace("http://", "").replace("https://", "") for l in page.links if l.url[:4] == "http"]
	except UnicodeDecodeError:
		return []
	return link_list

def lookup_domain_name(links_df, domain_url):
	"""Returns list of all URLs within domain web site (in database)"""
	return links_df[links_df['domain'] == domain_url]['url'].tolist()


##-----------------------------------------------------------
##-------- String Functions ---------------------------------
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

def build_url_feature_vector(A, search_list, string_to_search, reward_domain_set):
	"""Presence of search_list words in string,
	along with length of string"""
	feature_vector = check_strings(A, search_list, string_to_search)
	feature_vector.append(len(string_to_search))
	if string_to_search in reward_domain_set:
		feature_vector.append(1)
	else:
		feature_vector.append(0)
	return feature_vector


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

	def __init__(self, load_buffer=False, sample_size=50,
		save_file="data/buffer_data/replay_buffer.pickle"):
		self.save_file = save_file
		self.sample_size = sample_size

		if load_buffer == True:
			with open(self.save_file, 'rb') as f:
				self.buffer = pickle.load(f)
		else:
			self.buffer = OrderedDict()
			self.buffer['state'] = []
			self.buffer['next_state'] = []
			self.buffer['reward'] = []
			self.buffer['is_terminal'] = []

	def update(self, s, next_s, r, is_terminal):
		"""Update buffer with new transition (list of tuples)"""
		self.buffer['state'].append(s)
		self.buffer['next_state'].append(next_s)
		self.buffer['reward'].append(r)
		self.buffer['is_terminal'].append(is_terminal)

	def save(self):
		"""Write buffer to CSV file"""
		with open(self.save_file, 'wb') as f:
			pickle.dump(self.buffer, f, protocol=pickle.HIGHEST_PROTOCOL)

	def sample(self):
		"""Return a sample (dictionary) from the buffer"""
		buffer_size = len(self.buffer['reward'])
		sample_indices = random.sample(range(buffer_size), self.sample_size)
		sample_dict = {
			'state': np.array(self.buffer['state'])[sample_indices],
			'next_state': np.array(self.buffer['next_state'])[sample_indices],
			'reward': np.array(self.buffer['reward'])[sample_indices],
			'is_terminal': np.array(self.buffer['is_terminal'])[sample_indices]
		}
		return sample_dict

##-----------------------------------------------------------
##-------- DQN Agent ----------------------------------------
class CrawlerAgent(object):

	def __init__(self, url_list, reward_urls, word_list,
		cycle_freq, num_steps, print_freq, gamma=0.99, load_buffer=False,
		learning_rate=0.01,
		train_save_location="results/dqn_crawler_train_results_retry.csv",
		# train_save_location="results/dqn_crawler_train_results_retry_again.csv",
		tf_model_folder="models/linear_model"):

		# Set up state space and training parameters
		self.url_list = url_list
		self.reward_urls = reward_urls
		self.cycle_freq = cycle_freq
		self.num_steps = num_steps
		self.print_freq = print_freq
		self.gamma = gamma  # discount factor
		self.learning_rate = learning_rate
		self.words = word_list
		self.A = init_automaton(self.words)
		self.A.make_automaton()

		# Tensorflow placeholders and initialize buffer
		self.state = tf.placeholder(dtype=tf.float32, shape=[None, len(self.words)+2])
		self.next_state = tf.placeholder(dtype=tf.float32, shape=[None, len(self.words)+2])
		self.reward = tf.placeholder(dtype=tf.float32)
		self.is_terminal = tf.placeholder(dtype=tf.float32)
		self.build_target_net()
		self.replay_buffer = Buffer(load_buffer=load_buffer)

		# Set up train results dict
		self.train_results_dict = OrderedDict()
		self.train_results_dict['pages_crawled'] = []
		self.train_results_dict['total_reward'] = []
		self.train_results_dict['terminal_states'] = []
		self.train_results_dict['nn_loss'] = []
		self.train_save_location = train_save_location
		self.tf_model_folder = tf_model_folder

	def build_target_net(self):
		"""Build TF target network"""
		# idx = tf.where(tf.not_equal(self.state, 0))
		# sparse_state = tf.SparseTensor(idx, tf.gather_nd(self.state, idx), self.state.get_shape())
		# self.v = sparse_tensor_dense_matmul(sparse_state, self.weights)
		self.weights = tf.get_variable("weights", [len(self.words)+2, 1], 
			initializer = tf.random_normal_initializer(mean=0.0, stddev=0.001))
		self.bias = tf.get_variable('bias', [1], initializer = tf.constant_initializer(0.001))
		self.v = tf.matmul(self.state, self.weights) + self.bias
		self.v_next = tf.matmul(self.next_state, self.weights) + self.bias
		self.target = self.reward + (1-self.is_terminal) * self.gamma * tf.stop_gradient(tf.reduce_max(self.v_next))
		self.loss = tf.square(self.target - self.v)/2
		self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
		
	def save_train_results(self):
		"""Save train_results_dict as a Pandas dataframe"""
		train_results_df = pd.DataFrame(self.train_results_dict)
		train_results_df.to_csv(self.train_save_location, index=False, header=True)

	def save_tf_model(self, tf_session, tf_saver):
		tf_saver.save(tf_session, "/".join([self.tf_model_folder, "tf_model"]))


##-----------------------------------------------------------
##-----------------------------------------------------------
def main():
	##-------------------- Parameters
	cycle_freq = 50
	num_steps = 50000  # no. crawled pages before stopping
	print_freq = 1000
	epsilon = 0.05
	gamma = 0.5
	buffer_save_freq = 1000
	load_buffer = False
	learning_rate = 0.1
	reload_model = False

	##-------------------- Read in data
	# Company i.e. reward URLs
	companies_df = pd.read_csv('../data/domains_clean.csv')
	companies_df = companies_df[companies_df['vert_code'] <= 69203]
	companies_df = companies_df[companies_df['vert_code'] >= 69101]
	reward_urls = companies_df['url'].tolist()
	reward_urls = [l.replace("http://", "").replace("https://", "").replace("www.", "") for l in reward_urls]
	A_company = init_automaton(reward_urls)  # Aho-corasick automaton for companies
	A_company.make_automaton()

	# Rest of URLs to form the state space (remove any pages that obviously won't have hyperlinks/rewards)
	links_df = pd.read_csv('data/links_dataframe.csv')
	url_list = links_df['url'].tolist()
	url_list = [l.replace("http://", "").replace("https://", "") for l in url_list if type(l) is str if l[-4:] not in [".png", ".jpg", ".pdf", ".txt"]]
	url_set = set(url_list)

	# Load list of keywords
	words_list = read_csv_to_list('data/word_feature_list.csv')
	words_list = [w for w in words_list if w not in stops if len(w) > 1]
	url_endings_list = read_csv_to_list('data/domains_endings.csv')
	words_list = words_list + url_endings_list
	del url_endings_list

	##------------------- Initialize Crawler Agent and TF graph/session
	step_count = 0; pages_crawled = 0; total_reward = 0; terminal_states = 0
	reward_pages = []; recent_urls = [];
	reward_domain_set = set()

	tf.reset_default_graph()
	agent = CrawlerAgent(url_list, reward_urls, words_list, cycle_freq=cycle_freq, 
		num_steps=num_steps, print_freq=print_freq, gamma=gamma, load_buffer=load_buffer, learning_rate=learning_rate)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(init)

		if reload_model == True:
			print("Reloading model...")
			saver = tf.train.import_meta_graph('models/linear_model/tf_model.meta')
			saver.restore(sess, tf.train.latest_checkpoint('models/linear_model/'))
			all_vars = tf.get_collection('vars')
			weights_df = pd.DataFrame.from_dict({'words':words_list+['url_length', 'previous_reward'], 
				'coef': agent.weights.eval().reshape(-1).tolist()})
			weights_df.to_csv("results/feature_coefficients.csv", index=False, header=True)

			# # Test a URL
			# test_url = "www.tax.service.gov.uk"                                                                                              
			# # test_url = "www.panasonic.com"
			# s = np.array(build_url_feature_vector(agent.A, agent.words, test_url))
			# v = sess.run(agent.v, feed_dict={agent.state: s.reshape(1, -1)})
			# print("Value of URL {} is {}".format(test_url, float(v)))

		else:
			##------------------ Run and train crawler agent -----------------------
			while step_count < agent.num_steps:
				url = random.choice(list(url_set - set(recent_urls)))  # don't start at recent URL

				while step_count < agent.num_steps:
					step_count += 1

					# Keep track of recent URLs (to avoid loops)
					recent_urls.append(url)
					if len(recent_urls) > agent.cycle_freq:
						recent_urls = recent_urls[-agent.cycle_freq:]

					# Get rewards and remove domain if reward
					r, reward_url_idx = get_reward(url, A_company, reward_urls)
					pages_crawled += 1
					total_reward += r
					agent.train_results_dict['total_reward'].append(total_reward)
					if r > 0:
						reward_pages.append(url)
						reward_domain_set.update(lookup_domain_name(links_df, reward_urls[reward_url_idx]))
						reward_urls.pop(reward_url_idx)
						A_company = init_automaton(reward_urls)  # Aho-corasick automaton for companies
						A_company.make_automaton()
						# removed_reward_indices.append(reward_url_idx)		
						# url_set = url_set - reward_domain_set
					
					# Feature representation of current page (state) and links in page
					state = np.array(build_url_feature_vector(agent.A, agent.words, url, reward_domain_set)).reshape(1, -1)
					link_list = get_list_of_links(url)
					link_list = set(link_list).intersection(url_set)
					link_list = list(link_list - set(recent_urls))

					# Check if terminal state
					if r > 0 or len(link_list) == 0:
						terminal_states += 1
						is_terminal = 1
						next_state_array = np.zeros(shape=(1, len(words_list)+2))  # doesn't matter what this is
					else:
						is_terminal = 0
						next_state_list = [np.array(build_url_feature_vector(agent.A, agent.words, l, reward_domain_set)) for l in link_list]
						next_state_array = np.array(next_state_list)
					agent.train_results_dict['terminal_states'].append(terminal_states)

					# Train DQN
					train_dict = {
							agent.state: state,
							agent.next_state: next_state_array, 
							agent.reward: r, 
							agent.is_terminal: is_terminal
					}
					opt, loss, v_next, v  = sess.run([agent.opt, agent.loss, agent.v_next, agent.v], feed_dict=train_dict)
					agent.train_results_dict['nn_loss'].append(float(loss))

					# # Update buffer
					# agent.replay_buffer.update(state, next_state_array, r, is_terminal)
					# if step_count % buffer_save_freq == 0:
					# 	agent.replay_buffer.save()

					# Print progress + for debugging check the value function actually changes
					if step_count == 1:
						start_url = url
						start_state = state
						print("{} has value {}".format(start_url, float(v)))

					progress_bar(step_count+1, agent.num_steps)
					if step_count % agent.print_freq == 0:
						print("\nCrawled {} pages, total reward = {}, # terminal states = {}, remaining rewards = {}"\
						.format(pages_crawled, total_reward, terminal_states, len(reward_urls)))
					agent.train_results_dict['pages_crawled'].append(pages_crawled)

					# Choose next URL (and check for looping)
					if is_terminal == 1:
						break
					a = epsilon_greedy(epsilon, v_next)
					url = link_list[a]
			##-------------------------------------------------------------------------

			print("\nCrawled {} pages, total reward = {}, # terminal states = {}"\
				.format(pages_crawled, total_reward, terminal_states))
			agent.save_train_results()
			agent.save_tf_model(sess, saver)

			df = pd.DataFrame(reward_pages, columns=["rewards_pages"])
			df.to_csv('results/reward_pages.csv', index=False)

			v = sess.run(agent.v, feed_dict={agent.state: start_state.reshape(1, -1)})
			print("{} now has value {}".format(start_url, float(v)))

	sess.close()



if __name__ == "__main__":
	main()