#!/usr/bin/env python3
import lmdb
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("append_this_data")
	parser.add_argument("to_this_data")
	args = parser.parse_args()

	lst_data = []
	env = lmdb.open(args.append_this_data, readonly=True);
	with env.begin() as txn:
		cursor = txn.cursor();
		for key, value in cursor:
			innerlst_data = [key,value];
			lst_data.append(innerlst_data);

	env1 = lmdb.open(args.to_this_data, map_size=1024**4);
	with env1.begin(write=True) as txn1:
		for i in range(len(lst_data)):
			str_id = '{:08}'.format(i);
			txn1.put(lst_data[i][0] ,lst_data[i][1]);

if __name__ == "__main__":
	main()