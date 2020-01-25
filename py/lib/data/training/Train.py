import os
import tensorflow as tf
import tensorflow_datasets as tfds
from ..models import Modeler as Modeler
import math
import json
AUTOTUNE = tf.data.experimental.AUTOTUNE

class Train:
	def __init__(self, config, datasets):
		self.datasets = datasets
		self.config = config

	def begin_training(self, dataset, with_summary=False):
		print('...begin training')
		counts = dataset['counts']
		buffer_size = int(self.config['training']['buffer_size'])

		labels_out = self.write_labels(dataset)
		checkpoint_path = self.get_checkpoint_path(dataset)
		epochs = self.determine_epochs()

		shuffled, steps_per_epoch = self.shuffle(dataset, buffer_size=buffer_size)
		num_parallel_calls = int(self.config['training']['cores'])
		normalized = shuffled.map(self.normalize, num_parallel_calls=num_parallel_calls)
		fit = self.fit(normalized, checkpoint_path, counts, epochs=epochs, steps_per_epoch=steps_per_epoch)
		exported = self.export_trained_model(fit, dataset['reqs']['export_path'])
		return fit

	def export_trained_model(self, model, path):
		print('...exporting trained model', model, path)
		return tf.saved_model.save(model, path)

	def get_checkpoint_path(self, dataset):
		checkpoint_base = dataset['reqs']['checkpoints']
		checkpoint_path = os.path.join(checkpoint_base, dataset['name'])
		
		if not os.path.exists(checkpoint_path):
			created = os.mkdir(checkpoint_path)
		
		return checkpoint_path

	def create_export_model_location(self, dataset_name, piece):
		print('...creating export location')
		export_base = self.config['models']['tuned']
		export_path = os.path.join(export_base, dataset_name, piece)

		if not os.path.exists(export_path):
			created = os.makedirs(export_path, exist_ok=True)

		return export_path

	def create_training_dirs(self, dataset_name, piece):
		print('...creating checkpoint dir')
		reqs = ['checkpoints', 'cache', 'labels',]
		base_dir = os.path.join(self.config['training']['train'], dataset_name)
		training_dir = os.path.join(base_dir, piece)
		
		if not os.path.exists(training_dir):
			os.makedirs(training_dir, exist_ok=True)

		paths = {}
		for req in reqs:
			req_path = os.path.join(training_dir, req)
			paths[req] = req_path
			if not os.path.exists(req_path):
				os.mkdir(req_path)

		export_path = self.create_export_model_location(dataset_name, piece)
		paths['export_path'] = export_path
		return paths

	def determine_epochs(self, passes=9999999999):
		epochs = 1 * passes
		return epochs

	def determine_steps(self, set_size, batch_size):
		steps = math.ceil(set_size/batch_size)
		return steps

	def fit(self, dataset, checkpoint_path, counts, epochs=1, steps_per_epoch=10000):
		print("...begin fitting") 
		model = Modeler.Modeler(self.config)
		print("...epochs: {}, steps: {}, total: {}".format(epochs, steps_per_epoch, epochs*steps_per_epoch))
		fit = model.fit(dataset, checkpoint_path, counts, epochs=epochs, steps_per_epoch=steps_per_epoch)
		return fit

	@tf.function
	def normalize(self, image, label):
		print('...preprocessing dataset')
		return 2*image-1, label

	def run_training(self):
		for data_source, dataset in self.datasets.items():
			for set_piece in dataset:
				reqs = self.create_training_dirs(data_source, set_piece['name'])		
				reqs['export_path'] = self.create_export_model_location(data_source, set_piece['name'])
				set_piece['reqs'] = reqs
				trained = self.begin_training(set_piece, with_summary=True)

	def shuffle(self, dataset, buffer_size=10240):
		print('...prep for shuffle:', dataset['train'])
		unshuffled = dataset['train']
		num_images = dataset['counts']['image']
		batch_size = int(self.config['training']['batch_size'])
		
		print("...determine steps per epoch")
		steps_per_epoch = self.determine_steps(num_images, batch_size)

		print('...determing buffer size')
		if num_images < buffer_size:
			buffer_size = num_images
		
		print("...buffer size: {}".format(buffer_size))
		shuffled = unshuffled.shuffle(buffer_size).batch(batch_size)
		shuffled = shuffled.repeat()
		shuffled = shuffled.prefetch(buffer_size=AUTOTUNE)

		print("...shuffling")
		return shuffled, steps_per_epoch

	def write_labels(self, dataset):
		raw_label_map = dataset['raw']['label']
		outfile_base = dataset['reqs']['labels']
		outfile_name = dataset['name'] + '.pbtxt'
		outfile_path = os.path.join(outfile_base, outfile_name)
		print("...writing labels map to", outfile_path)
		json.dump(raw_label_map, open(outfile_path, 'w'))
		return

