"""
File Reader API for external clustering benchmark.
reads the bin data and converts it to data and label np array
"""
import numpy as np
import zlib
import json
import os

def read_dataset(name):
	"""
	returns data and label np array having the name
	"""
	path = "./compressed/" + name + "/"
	path_data = path + "data.bin"
	path_labels = path + "labels.bin"
	## open the data and label binary file
	with open(path_data, 'rb') as f:
		data_comp = f.read()
	with open(path_labels, 'rb') as f:
		labels_comp = f.read()
	## convert the data and label to np array
	data = np.array(json.loads(zlib.decompress(data_comp).decode('utf8')))
	labels = np.array(json.loads(zlib.decompress(labels_comp).decode('utf8')))

	return data, labels

def read_dataset_by_path(path):
	path_data = path + "data.bin"
	path_labels = path + "labels.bin"
	## open the data and label binary file
	with open(path_data, 'rb') as f:
		data_comp = f.read()
	with open(path_labels, 'rb') as f:
		labels_comp = f.read()
	## convert the data and label to np array
	data = np.array(json.loads(zlib.decompress(data_comp).decode('utf8')))
	labels = np.array(json.loads(zlib.decompress(labels_comp).decode('utf8')))

	return data, labels

def read_multiple_datasets(names):
	data = {}
	labels = {}
	for name in names:
		data_, labels_ = read_dataset(name)
		data[name] = data_
		labels[name] = labels_
	
	return data, labels

def read_all_datasets():
	names = os.listdir("./compressed/")
	return read_multiple_datasets(names)