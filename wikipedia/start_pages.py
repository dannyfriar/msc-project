import sys
import pickle
import ujson as json

from urllib.request import urlopen

title_blacklist = ['User', 'Talk', 'Wikipedia', 'Help', 'Template', 'Category', 'Portal', 'File']
url_base = ''.join(['https://en.wikipedia.org/w/api.php?action=query&titles={}',
	'&prop=links&pllimit=max&format=json'])

def progress_bar(value, endvalue, bar_length=20):
	percent = float(value) / endvalue
	arrow = '-' * int(round(percent * bar_length)-1) + '>'
	spaces = ' ' * (bar_length - len(arrow))
	sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
	sys.stdout.flush()

def title_in_blacklist(title):
	"""Check if title contains words in blacklist"""
	if sum([t in title for t in title_blacklist]) > 0:
		return True
	return False

def get_wiki_links(url_title):
	"""Gets wikipedia URLs given page title"""
	url = url_base.format(url_title)
	url = url.replace(' ', '%20')
	try:
		response = urlopen(url).read().decode('UTF-8')
	except UnicodeEncodeError as e:
		return []
	response = json.loads(response)['query']['pages']
	pages_dict = response[next(iter(response))]
	try:
		links = pages_dict['links']
		title_links_list = [d['title'] for d in links]
		title_links_list = [t for t in title_links_list if not title_in_blacklist(t)]
	except KeyError as e:
		title_links_list = []
	return title_links_list

# Get pages one hop from philosophy
print("Getting links from philosophy page...")
one_hop_pages = get_wiki_links('Philosophy')
with open('first_hops.pkl', 'wb') as f:
	pickle.dump(one_hop_pages, f)

# One hop from these pages
print("Getting links from these pages...")
two_hop_pages = []
for i, title in enumerate(one_hop_pages):
	progress_bar(i+1, len(one_hop_pages))
	title_links_list = get_wiki_links(title)
	two_hop_pages += title_links_list

with open('second_hops.pkl', 'wb') as f:
	pickle.dump(two_hop_pages, f)
