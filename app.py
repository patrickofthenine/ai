import os
import sys
import importlib
import py.lib.utils.config_util 	as ConfigUtil
import py.lib.utils.data_util 		as DataUtil
import py.lib.devices.Device 		as Device
import py.lib.data.datasets.Builder as Builder
import py.lib.data.training.Train 	as Train
import tensorflow 					as tf

ROOT_DIR = os.path.abspath(os.curdir)

class App:
	def __init__(self, name):
		self.name = name
		self.root = ROOT_DIR
		self.config = self.get_app_config()
		return
	
	def build_datasets(self):
		builder = Builder.Builder(self.config)	
		datasets = builder.run_builder(guess_type=False)
		return datasets

	def detect_devices(self, with_benchmark):
		device_config = ConfigUtil.ConfigUtil(ROOT_DIR).get_config('configs/devices.yml')
		D = Device.Device(device_config)
		devices = D.detect_devices(with_benchmark=with_benchmark)

	def get_app_config(self):
		self.config = ConfigUtil.ConfigUtil(ROOT_DIR).get_config('configs/app.yml')[self.name]
		return self.config

	def get_datasets(self):
		data_util = DataUtil.DataUtil(self.config, 'datasets')
		data_util.get_and_expand()	
		return 

	def get_models(self):
		data_util = DataUtil.DataUtil(self.config, 'models')
		data_util.get_and_expand()
		data_util.source_models()
		return 

	def run_training(self, built_datasets):
		T = Train.Train(self.config, built_datasets).run_training()
		return T 

	def run(self, with_benchmark=False):
		devices = self.detect_devices(with_benchmark=with_benchmark)
		models = self.get_models()
		datasets = self.get_datasets()
		built_datasets = self.build_datasets()	
		trained = self.run_training(built_datasets)
		return 

A = App('AI').run(with_benchmark=False)
