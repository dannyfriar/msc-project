import time
import gensim
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer

from evolutionai import StorageEngine
storage = StorageEngine("/nvme/rl_project_webcache/")

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

# Anchor text
text_word_list = pd.read_csv("new_data/all_vocab.csv")['word'].tolist()
text_word_dict = dict(zip(text_word_list, list(range(len(text_word_list)))))
text_count_vec = CountVectorizer(vocabulary=text_word_dict)

# URLs
# words_list = pd.read_csv("data/new_segmented_words_df.csv")['word'].tolist()
words_list = pd.read_csv("data/embedding_segmented_words_df.csv")['word'].tolist()
word_dict = dict(zip(words_list, list(range(len(words_list)))))
count_vec = CountVectorizer(vocabulary=word_dict)

# URL state space
links_df = pd.read_csv("new_data/links_dataframe.csv")
rm_list = ['aarp.org', 'akc.org', 'alcon.com', 'lincoln.com', 'orlakiely.com', 
'red.com', 'ef.com', 'ozarksfirst.com']
links_df['domain'] = links_df.domain.str.replace("www.", "")
links_df = links_df[~links_df['domain'].isin(rm_list)]
url_list = links_df['url'].tolist()

#------ Load Word2Vec and create embeddings matrix from vocabulary
print("Loading word2vec...")
model = gensim.models.Word2Vec.load_word2vec_format('../../GoogleNews-vectors-negative300.bin', binary=True)
print(model['queen'])
embeddings = np.zeros((len(text_word_list), len(model['queen'])))

errors = 0
error_keys = []
for k, v in text_word_dict.items():
	try:
		embeddings[v] = model[k]
	except KeyError:
		errors += 1
		error_keys.append(k)
		pass

anchor_word_list = [w for w in text_word_list if w not in error_keys]
pd.DataFrame.from_dict({'word':anchor_word_list}).to_csv("../../embedded_anchor_list.csv")
print("Number of errors = {}".format(errors))
print("Saving embeddings matrix...")
np.savetxt('../../anchor_embeddings_matrix.csv', embeddings, delimiter=',')

# #------ Create URL embeddings matrix
# print("Loading word2vec...")
# model = gensim.models.Word2Vec.load_word2vec_format('../../GoogleNews-vectors-negative300.bin', binary=True)
# print(model['queen'])
# embeddings = np.zeros((len(words_list), len(model['queen'])))

# errors = 0
# error_keys = []
# for k, v in word_dict.items():
# 	try:
# 		embeddings[v] = model[k]
# 	except KeyError:
# 		errors += 1
# 		error_keys.append(k)
# 		pass

# url_word_list = [w for w in words_list if w not in error_keys]
# pd.DataFrame.from_dict({'word':url_word_list}).to_csv("../../embedded_url_word_list.csv")
# print("Number of errors = {}".format(errors))
# print("Saving URL embeddings matrix...")
# np.savetxt('../../url_embeddings_matrix.csv', embeddings, delimiter=',')


# # Test a random URL
# random.seed(12345)
# url = random.choice(url_list)
# link_list = get_list_of_links(url)

# # Embeddings matrix
# embeddings = np.loadtxt('../../url_embeddings_matrix.csv', delimiter=',')

# # URL words
# words_list = pd.read_csv("../../embedded_url_word_list.csv")['word'].tolist()
# word_dict = dict(zip(words_list, list(range(len(words_list)))))
# count_vec = CountVectorizer(vocabulary=word_dict)

# # Make matrix to feed into tensorflow for s
# url_feature_matrix = count_vec.transform([url]).toarray()
# s = np.nonzero(url_feature_matrix)[1]
# s = np.pad(s, (0, 10-len(s)), 'constant', constant_values=(0,0)).reshape(1, -1)
# print(s.shape)

# # Do the same for s'
# link_feature_matrix = count_vec.transform(link_list).toarray()
# s_prime = np.nonzero(link_feature_matrix)
# splits = np.cumsum(np.bincount(s_prime[0]))
# s_prime = np.split(s_prime[1], splits.tolist())[:-1]
# s_prime = [np.pad(s, (0, 10-len(s)), 'constant', constant_values=(0,0)) for s in s_prime]
# s_prime = np.stack(s_prime, axis=0)
# print(s_prime.shape)

# def conv2D(x, W, b, stride=2):
# 	"""Apply 2D conv + ReLU"""
# 	conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
# 	conv = tf.nn.relu(tf.nn.bias_add(conv, b))
# 	return conv


# filter_sizes = [1, 2, 3]
# num_filters = 3
# state = tf.placeholder(dtype=tf.int32, shape=[None, 10])
# next_state = tf.placeholder(dtype=tf.int32, shape=[None, 10])

# embedded_state = tf.expand_dims(tf.nn.embedding_lookup(embeddings, state), -1)
# embedded_next_state = tf.expand_dims(tf.nn.embedding_lookup(embeddings, next_state), -1)

# pooled_outputs = []
# for i, filter_size in enumerate(filter_sizes):
# 	with tf.name_scope("conv-maxpool-%s" % filter_size):
# 		# Convolution Layer
# 		filter_shape = [filter_size, 300, 1, num_filters]
# 		W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
# 		b = tf.Variable(tf.constant(0.01, shape=[num_filters]), name="b")
# 		conv = tf.nn.conv2d(tf.cast(embedded_state, tf.float32), W, 
# 			strides=[1, 1, 1, 1], padding="VALID", name="conv")
# 		h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
# 		pooled = tf.nn.max_pool(h,ksize=[1, 10 - filter_size + 1, 1, 1],
# 			strides=[1, 1, 1, 1], padding='VALID', name="pool")
# 		pooled_outputs.append(pooled)

# num_filters_total = num_filters * len(filter_sizes)
# h_pool = tf.concat(pooled_outputs, 3)
# h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
# print(h_pool_flat.get_shape())


# with tf.Session() as sess:
# 	sess.run(init)
	# sess.close()




















