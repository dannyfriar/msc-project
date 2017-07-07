import os
import sys
import re
import csv
import time
import random
import argparse
import ahocorasick
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import OrderedDict

from evolutionai import StorageEngine
storage = StorageEngine("/nvme/rl_project_webcache/")


# Read in data
links_df = pd.read_csv("../new_data/links_dataframe.csv")
company_urls = links_df[links_df['type']=='company-url']['url']
company_urls = [l.replace("www.", "") for l in company_urls]
print(len(company_urls))

url_set = set(links_df['url'].tolist())
url_list = list(url_set)
print(len(url_list))