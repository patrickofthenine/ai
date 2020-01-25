import os
from .. import Builder as Builder
import tensorflow as tf
import time
import json
AUTOTUNE = tf.data.experimental.AUTOTUNE

class Image:
	def __init__(self, handler_config):
		self.handler_config = handler_config
		self.extensions = ['.png', '.jpg', '.jpeg', '.bmp']

	def _check_image_dataset_contents(self, dataset):
		plt.figure(figsize=(8,8))
		for n, image in enumerate(dataset.take(1)):
			print(n, image)

	def get_tensorflow_dataset(self, items, name):
		print('...creating tensorflow dataset from', len(items), name)
		dataset = tf.data.Dataset.from_tensor_slices(items)
		if name == 'images':
			dataset = dataset.map(self.load_and_preprocess_image)	
		return dataset

	def zip_datasets(self, set_a, set_b):
		zipped = tf.data.Dataset.zip( (set_a, set_b) )
		return zipped

	def get_labels_from_path(self, rel_path):
		# hsf_6/digit/33/33_00706.png
		bad_labels = self.handler_config['ignore_as_label']
		path = os.path.dirname(rel_path)	
		base, ext = os.path.splitext(rel_path) 
		path_parts = path.split(os.sep)
		valid_ext = ext in self.extensions

		good_labels = []
		if valid_ext:
			for part in path_parts:
				good_part = True
				for bad_label in bad_labels:
					if bad_label in part:
						good_part = False
	
				if good_part:
					good_labels.append(part)	
		return good_labels

	def create_raw_label_map(self, labels):
		label_map = {}
		for index, label in enumerate(labels):
			label_map[index] = label
		return label_map

	def create_image_label_dataset(self, content):
		print('...creating image_label_dataset')
		start = time.time()

		set_name = os.path.basename(content)
		images, labels, unique_labels = self.walk_content(content)
		raw_label_map = self.create_raw_label_map(unique_labels)
		labels_index = self.index_all_labels(labels, unique_labels)
		image_dataset = self.get_tensorflow_dataset(images, 'images')
		label_dataset = self.get_tensorflow_dataset(labels_index, 'labels')
		image_label_dataset = self.zip_datasets(image_dataset, label_dataset)

		if len(images) == len(labels):
			print('...image-label counts match:', len(images), len(labels))
	
		end = time.time()
		print("...run time:", end - start, 's')
		
		return {
			'name': set_name,
			'train': image_label_dataset,
			'data': {
				'image': image_dataset,
				'label': label_dataset,
			},
			'counts': {
				'image': len(images),
				'label': len(labels),
				'unique_labels': len(unique_labels)
			},
			'raw': {
				'label': raw_label_map
			}

		}

	def walk_content(self, from_path):
		print('...getting images, labels, unique_labels')
		images = []
		labels = []
		unique_labels = []

		#look if more efficient way than this built in os.walk
		for path, dirs, files in os.walk(from_path):
			for file in files:
				file_path = os.path.join(path, file)
				relative_path = os.path.relpath(file_path, from_path)
				files_labels = self.get_labels_from_path(relative_path)
				for label in files_labels:
					if not label in unique_labels:
						unique_labels.append(label)
					images.append(file_path)
					labels.append(label)
		return images, labels, unique_labels

	def index_all_labels(self, labels, unique_labels):
		print('...creating label index')
		label_index = []
		for label in labels:
			label_index.append(unique_labels.index(label))		
		return label_index

	def load_and_preprocess_image(self, path):
		image = tf.io.read_file(path)
		return self.preprocess_image(image)

	def preprocess_image(self, image):
		image = tf.io.decode_png(image, channels=3)
		image = tf.image.resize(image, [128,128])
		image = image / 255.0
		return image

	def run_builder(self, set_dir):
		print('...running image dataset builder from', set_dir)
		contents = os.listdir(set_dir)
		datasets = []
		for content in contents:
			ignore_during_build = self.handler_config['ignore_during_build']
			buildable = True
			for ignore in ignore_during_build:
				if ignore in content:
					buildable = False

			if buildable:
				content_dir = os.path.join(set_dir, content)
				content_is_directory = os.path.isdir(content_dir)
				if content_is_directory:
					print("\nSource:", content)
					print("...inspecting data source:", content)
					image_label_dataset = self.create_image_label_dataset(content_dir)
					datasets.append(image_label_dataset)
		return datasets