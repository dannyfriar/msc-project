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

def load_csv_to_list(file_name):
	"""Loads single column CSV as list"""
	with open(file_name) as f:
		reader = csv.reader(f)
		return list(reader)[0]


##-----------------------------------------------------------
##-------- Get links from LMDB data functions ---------------
def get_list_of_links(url, s=storage):
	"""Use the LMDB database to get a list of links for a given URL"""
	try:
		page = s.get_page(url)
	except UnicodeError:
		return []
	if page is None:
		return []
	try:
		link_list = [l.url for l in page.links if l.url[:4] == "http"]
		link_list = [l.replace("http://", "") for l in link_list]
		link_list = [l.replace("https://", "") for l in link_list]
	except UnicodeDecodeError:
		return []
	return link_list

def get_all_links(url_list):
	"""Get all links from a list of URLs"""
	full_link_list = []
	skipped_urls = []
	for idx, url in enumerate(url_list):
		# progress_bar(idx+1, len(url_list))
		try:
			link_list = get_list_of_links(url)
		except (UnicodeError, IndexError):
			skipped_urls.append(url)
		full_link_list = full_link_list + link_list
	full_link_list = full_link_list + url_list
	full_link_list = list(set(full_link_list))
	# print("\nSkipped %d URLs" % len(skipped_urls))
	return full_link_list


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

def build_url_feature_vector(A, search_list, string_to_search):
	"""Presence of search_list words in string,
	along with length of string"""
	feature_vector = check_strings(A, search_list, string_to_search)
	feature_vector.append(len(string_to_search))
	return feature_vector


##-----------------------------------------------------------
##-------- DQN Agent ----------------------------------------
class CrawlerAgent(object):

	def __init__(self, url_list, reward_urls, word_list,
		discount_factor=0.99,
		learning_rate=0.001,
		cycle_freq=10, num_steps=5000, print_freq=1000,
		train_save_location="results/dqn_crawler_train_results.csv",
		tf_model_folder="models"):

		# Set up state space
		self.url_list = url_list
		self.reward_urls = reward_urls
		self.cycle_freq = cycle_freq
		self.num_steps = num_steps
		self.print_freq = print_freq

		# Training parameters - RL and TF
		self.discount_factor = discount_factor
		self.learning_rate = learning_rate

		# Build search words
		self.words = word_list
		self.A = init_automaton(self.words)
		self.A.make_automaton()

		# Tensorflow placeholders
		self.state = tf.placeholder(dtype=tf.float32, shape=[None, len(self.words)+1])
		self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1])
		self.build_target_net()

		# Set up train results dict
		self.train_results_dict = OrderedDict()
		self.train_results_dict['pages_crawled'] = []
		self.train_results_dict['total_reward'] = []
		self.train_results_dict['terminal_states'] = []
		self.train_save_location = train_save_location
		self.tf_model_folder = tf_model_folder

	def build_target_net(self):
		"""Build TF target network"""
		# idx = tf.where(tf.not_equal(self.state, 0))
		# sparse_state = tf.SparseTensor(idx, tf.gather_nd(self.state, idx), self.state.get_shape())
		self.weights = tf.get_variable("weights", [len(self.words)+1, 1], initializer = tf.random_normal_initializer())
		# self.v = sparse_tensor_dense_matmul(sparse_state, self.weights)
		self.v = tf.sigmoid(tf.matmul(self.state, self.weights))

	def train_target_net(self):
		""""""
		pass

	def save_train_results(self):
		"""Save train_results_dict as a Pandas dataframe"""
		train_results_df = pd.DataFrame(self.train_results_dict)
		train_results_df.to_csv(self.train_save_location, index=False, header=True)


##-----------------------------------------------------------
##-------- RL Functions -------------------------------------
def get_reward(url, company_urls):
	"""Return 1 if company URL, 0 otherwise"""
	return 1 if url in company_urls else 0

def epsilon_greedy(epsilon, action_list):
	"""Returns index of chosen action"""
	if random.uniform(0, 1) > epsilon:
		return np.argmax(action_list)
	else:
		return random.randint(0, len(action_list)-1)


##-----------------------------------------------------------
##-----------------------------------------------------------
def main():
	##-------------------- Parameters
	cycle_freq = 5
	number_crawls = 10000  # no. crawled pages before stopping
	print_freq = 1000
	epsilon = 0.1

	##-------------------- Read in data
	# Company i.e. reward URLs
	companies_df = pd.read_csv('../data/domains_clean.csv')
	companies_df = companies_df[companies_df['vert_code'] <= 69203]
	companies_df = companies_df[companies_df['vert_code'] >= 69101]
	reward_urls = companies_df['url'].tolist()

	# Rest of URLs to form the state space
	first_hop_df = pd.read_csv('data/first_hop_links.csv', names = ["url"])
	url_list = reward_urls + first_hop_df['url'].tolist()
	second_hop_df = pd.read_csv('data/second_hop_links.csv', names = ["url"])
	url_list = url_list + second_hop_df['url'].tolist()
	del companies_df, first_hop_df, second_hop_df

	# Remove any pages that obviously won't have hyperlinks/rewards
	url_list = [l for l in url_list if l[-4:] not in [".png", ".jpg", ".pdf", ".txt"]]

	# Load vert words
	words_list = load_csv_to_list('../data/vert_desc_words.csv')
	# word_list = words.words() + names.words()
	words_list = [w for w in words_list if w not in stops]
	words_list = [w for w in words_list if len(w) > 1]

	##------------------- Initialize and train Crawler Agent
	step_count = 0
	pages_crawled = 0
	total_reward = 0
	terminal_states = 0
	reward_pages = []
	recent_urls = []

	##-----------------------
	with tf.device('/cpu:0'):
		tf.reset_default_graph()
		agent = CrawlerAgent(url_list, reward_urls, words_list)

		init = tf.global_variables_initializer()

		with tf.Session() as sess:
			sess.run(init)

			while step_count < agent.num_steps:
				url = random.choice([l for l in agent.url_list if l not in recent_urls])  # don't start at recent URL

				while step_count < agent.num_steps:
					step_count += 1

					# Track progress
					progress_bar(step_count+1, agent.num_steps)
					if step_count % agent.print_freq == 0:
						print("\nCrawled {} pages, total reward = {}, # terminal states = {}"\
						.format(pages_crawled, total_reward, terminal_states))
					agent.train_results_dict['pages_crawled'].append(pages_crawled)
					agent.train_results_dict['total_reward'].append(total_reward)
					agent.train_results_dict['terminal_states'].append(terminal_states)

					# Keep track of recent URLs (to avoid loops)
					recent_urls.append(url)
					if len(recent_urls) > agent.cycle_freq:
						recent_urls = recent_urls[-agent.cycle_freq:]

					# Get rewards
					r = get_reward(url, agent.reward_urls)
					pages_crawled += 1
					total_reward += r
					if r > 0:
						reward_pages.append(url)

					# Choose a next URL
					link_list = get_list_of_links(url)
					link_list = [l for l in link_list if l in agent.url_list if l not in recent_urls]

					if len(link_list) == 0:
						terminal_states += 1
						break

					next_state_list = [np.array(build_url_feature_vector(agent.A, agent.words, l)) for l in link_list]
					next_state_array = np.array(next_state_list)
					v = sess.run(agent.v, feed_dict={agent.state: next_state_array})
					a = epsilon_greedy(epsilon, v)
					url = link_list[a]

			print("\nCrawled {} pages, total reward = {}, # terminal states = {}"\
				.format(pages_crawled, total_reward, terminal_states))
			agent.save_train_results()

		sess.close()







if __name__ == "__main__":
	random.seed(1234)
	main()