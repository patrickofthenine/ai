import os
import sys
import pprint as pprint
pp = pprint.PrettyPrinter(indent=4)
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import glob
import pathlib
import operator
import time
from datetime import datetime
import matplotlib.pyplot as plt
from .builders import Audio, Image, Text, Video
from ....handlers import Handler
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

class Builder:
	def __init__(self, config):
		self.name = 'Dataset Sourcing and Building'
		self.config = config
		self.datatype_map = {
			'Audio': {
				'extensions': ['.mp3', '.wav'],
			},
			'Image':  {
				'extensions': ['.png', '.jpg', '.jpeg', '.bmp'],
			},
			'Text': {
				'extensions': ['.txt', '.csv'],
			},
			'Video': {
				'extensions': ['.mp4'],
			},
		}
	def get_builder_by_type(self, name, handler):
		if name == 'Audio':
			return Audio.Audio(handler)
		elif name == 'Image':
			return Image.Image(handler)
		elif name == 'Video':
			return Video.Video(handler)
		elif name == 'Text':
			return Text.Text(handler)
		else:
			return None

	def get_dataset_by_name(self, name):
		dataset = os.path.join(self.config['datasets']['expanded'], name)
		return dataset

	def get_dataset_list(self):
		dataset_list = os.listdir(self.config['datasets']['expanded'])
		dataset_list = list(filter(lambda file: not file.startswith('.'), dataset_list))
		return dataset_list

	def maybe_detect_dataset_type(self, dataset_path):
		print('...trying to detect dataset type')
		file_types = {}
		for path, dirs, files in os.walk(dataset_path):
			files = list(filter( lambda file: not file.startswith("."), files))
			for file in files:
				file_name, file_type = os.path.splitext(file)
				if file_type in file_types.keys():
					file_types[file_type] = file_types[file_type] + 1 
				else:
					file_types[file_type] = 0

		highest_item, highest_item_count = max(file_types.items(), key=operator.itemgetter(1))
		for data_type, tools in self.datatype_map.items():
			for tool in tools:
				if tool == 'extensions':
					in_list = highest_item in tools[tool]
					if in_list:
						return data_type
		return None

	def run_builder(self, guess_type=False):
		print("\nRunning Builder. Guess types:", guess_type)
		datasets = self.get_dataset_list()
		dataset_count = len(datasets)
		built = {}
		print("...sets in builder:", dataset_count, datasets)
		for dataset in datasets:
			if dataset == 'NIST':
				print("...dataset:", dataset)
				dataset_dir = self.get_dataset_by_name(dataset)
				if guess_type:
					print('...attempting to guess data type')
					dataset_type = self.maybe_detect_dataset_type(dataset_dir)
					print('...got:', dataset_type)
				else:
					dataset_type = 'Image'

				print('...builder type:', dataset_type)
				set_handler = Handler.Handler().get_handler(dataset)	
				builder = self.get_builder_by_type(dataset_type, set_handler)
				built_dataset = builder.run_builder(dataset_dir)
				built[dataset] = built_dataset
		print("\nBuilt sets:", len(built.items()))
		return built
