import os
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
tf.executing_eagerly()

class Modeler:
	def __init__(self, config):
		self.name = "Model"
		self.config = config
		self.base_model = self.get_pretrained_model()

	def layer(self, base, counts):
		print('...layering')

		model = tf.keras.Sequential([
			base,
			tf.keras.layers.GlobalAveragePooling2D(),
			tf.keras.layers.Dense(counts['unique_labels'])
		])

		print('...compiling\n')
		model.compile(
			optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy']
		)
		self.summarize(model)
		return model

	def evaluate(self, model, data, steps):
		print('...evaluating', model, steps)
		loss, acc = model.evaluate(data, steps=steps)
		print("Loss: {}, Accuracy: {}".format(loss, acc))
		return loss, acc

	def save_checkpoint(self, path):
		return tf.keras.callbacks.ModelCheckpoint(
			filepath=path,
			save_weights_only=True,
			verbose=1
		)
	
	def fit(self, shuffled, checkpoint_path, counts, epochs=3, steps_per_epoch=100):
		print("...getting layered model")	
		model = self.layer(self.base_model, counts)
		save_checkpoint = self.save_checkpoint(checkpoint_path)
		callbacks = [save_checkpoint]

		model.fit(shuffled, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
		return model

	def get_models(self):
		print('...getting models list')
		path_to_models = self.config['models']['expanded']
		models_list = list(filter(lambda p: os.path.isdir(os.path.join(path_to_models, p)), os.listdir(path_to_models)))	
		return

	def get_pretrained_model(self, shape=(128, 128, 3)):
		print('...getting pretrained model')
		pretrained_model = tf.keras.applications.MobileNetV2(input_shape=shape, include_top=False, weights='imagenet')
		pretrained_model.trainable=False
		return pretrained_model

	def summarize(self, model):
		return model.summary()

	def save_model(self, model, path):
		print("SAVE IT", model, path)
		return