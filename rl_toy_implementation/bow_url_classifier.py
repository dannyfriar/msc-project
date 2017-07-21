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

RESULTS_FOLDER = "results/bow_classifier_results/"
MODEL_FOLDER = "models/"

##-----------------------------------------------------------
##-------- Miscellaneous Functions --------------------------
def progress_bar(value, endvalue, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


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
		print("In exception!!")
		s_out = s[:max_len]
	return s_out

def build_url_feature_matrix(count_vec, url_list):
	"""Return 2d numpy array of booleans"""
	feature_matrix = count_vec.transform(url_list).toarray()
	return feature_matrix

##-----------------------------------------------------------
##-------- Classifier ---------------------------------------
class CNNClassifier(object):
	def __init__(self, weights_shape, tf_model_folder, learning_rate):
		self.weights_shape = weights_shape
		self.learning_rate = learning_rate
		self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.weights_shape])
		self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
		self.reg_net()
		self.tf_model_folder = tf_model_folder

	def reg_net(self):
		self.weights = tf.get_variable("weights", [self.weights_shape, 1], 
			initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001))
		self.bias = tf.get_variable('bias', [1], initializer=tf.constant_initializer(0.001))
		self.x_out = tf.matmul(self.x, self.weights) + self.bias
		self.loss = tf.square(self.y - self.x_out) / 2
		self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

	def save_tf_model(self, tf_session, tf_saver):
		tf_saver.save(tf_session, "/".join([self.tf_model_folder, "tf_model"]))


##-----------------------------------------------------------
##-----------------------------------------------------------
def main():
	##-------------------- Parameters
	learning_rate = 0.0005
	reload_model = False
	max_len = 50
	embedding_size = 300
	filter_sizes = [1, 2, 3, 4]
	num_filters = 4
	batch_size = 1000
	val_batch_size = 1000
	n_batches = 2000
	print_freq = 100
	reload_model = True

	##-------------------- Read in data
	print("Loading Data...")
	links_df = pd.read_csv("new_data/links_dataframe.csv")
	rm_list = ['aarp.org', 'akc.org', 'alcon.com', 'lincoln.com', 'orlakiely.com', 
	'red.com', 'ef.com', 'ozarksfirst.com']  # remove mis-labelled reward URLs
	links_df['domain'] = links_df.domain.str.replace("www.", "")
	links_df = links_df[~links_df['domain'].isin(rm_list)]
	links_df['value'] = 0
	links_df.loc[links_df['type'] == 'company-url', 'value'] = 1
	links_df.loc[links_df['type'] == 'first-hop-link', 'value'] = 0.75
	links_df.loc[links_df['type'] == 'second-hop-link', 'value'] = 0.75**2
	links_df = shuffle(links_df)
	train, test = train_test_split(links_df, test_size=0.5)
	validation, test = train_test_split(test, test_size=0.8)

	# Rebalancing data between classes in training set
	company_df_list = [train[train['type'] == 'company-url']]*200
	first_links_df_list = [train[train['type'] == 'first-hop-link']]*6
	df_list = company_df_list + first_links_df_list
	df_list.append(train)
	train = shuffle(pd.concat(df_list))

	# Embeddings matrix
	print("Loading words and embeddings...")
	embeddings = np.loadtxt('../../url_embeddings_matrix.csv', delimiter=',')

	# URL words
	words_list = pd.read_csv("../../embedded_url_word_list.csv")['word'].tolist()
	word_dict = dict(zip(words_list, list(range(len(words_list)))))
	count_vec = CountVectorizer(vocabulary=word_dict)
	weights_shape = len(words_list)

	# File locations
	training_file = RESULTS_FOLDER + "all_urls_revisit.csv"
	model_save_file = MODEL_FOLDER + "bow_classifier"
	test_value_files = RESULTS_FOLDER + "test_value.csv"
	feature_coefs_save_file = RESULTS_FOLDER + "feature_coefs.csv"
	loss_list = []; batch_count = 0

	##------------------- Initialize TF graph/session
	tf.reset_default_graph()
	agent = CNNClassifier(weights_shape, model_save_file, learning_rate=learning_rate)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(init)

		if reload_model == True:
			print("Reloading model...")
			saver = tf.train.import_meta_graph(model_save_file+"/tf_model.meta")
			saver.restore(sess, tf.train.latest_checkpoint(model_save_file))
			all_vars = tf.get_collection('vars')
			weights_df = pd.DataFrame.from_dict({'words':words_list, 'coef': agent.weights.eval().reshape(-1).tolist()})
			weights_df.to_csv(feature_coefs_save_file, index=False, header=True)

			test_batch = validation.sample(n=20000)
			urls_test = test_batch['url'].tolist()
			url_test_array = build_url_feature_matrix(count_vec, urls_test)
			output = sess.run(agent.x_out, feed_dict={agent.x: url_test_array})
			test_batch['predicted_value'] = output.reshape(-1).tolist()
			test_batch.to_csv(test_value_files, index=False)

		else:
			print("Training...")
			if os.path.isfile(training_file):
				os.remove(training_file)

			for i in range(n_batches):
				# Get batch of URLs
				batch_df = train.sample(n=batch_size)
				y = np.array(batch_df['value'].tolist()).reshape(-1, 1)
				urls = batch_df['url'].tolist()

				# Build URL array and train classifier
				url_array = build_url_feature_matrix(count_vec, urls)
				opt, loss = sess.run([agent.opt, agent.loss], feed_dict={agent.x: url_array, agent.y: y})
				loss = float(np.mean(loss))
				loss_list.append(loss)

				# progress_bar(batch_count, n_batches)

				if batch_count % print_freq == 0:
					batch_val_df = validation.sample(n=val_batch_size)
					y_val = np.array(batch_val_df['value'].tolist()).reshape(-1, 1)
					urls_val = batch_val_df['url'].tolist()
					url_val_array = build_url_feature_matrix(count_vec, urls_val)
					val_loss = sess.run([agent.loss], feed_dict={agent.x: url_val_array, agent.y: y_val})
					val_loss = np.mean(val_loss)

					with open(training_file, "a") as csv_file:
						writer = csv.writer(csv_file, delimiter=',')
						writer.writerow([batch_count, loss, val_loss])

					print("\nAfter {} batches, training loss = {}, validation loss = {}"\
						.format(batch_count, loss, val_loss))
					agent.save_tf_model(sess, saver)
				batch_count += 1

	sess.close()


if __name__ == "__main__":
	main()