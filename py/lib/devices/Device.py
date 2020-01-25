import os
import sys
import tensorflow as tf
import time
import pprint
from ..utils.benchmarks import Benchmark as Benchmark
import importlib.util 
pp = pprint.PrettyPrinter()

class Device:
	def __init__(self, config):
		self.name = 'Device'
		self.device_types = ['CPU', 'GPU']
		self.config = config[self.name]
		self.devices = self.detect_devices()
		return 
	
	def detect_devices(self, provided_types=None, only_provided=False, with_benchmark=False):
		if only_provided: 
			if isinstance(only_provided, list):
				types = provided_types
			else:
				types = []
				for provided_type in provided_types:
					if isinstance(provided_type, str):
						types.append(provided_type)
		else:
			if isinstance(provided_types, list):
				types = self.device_types + provided_types
			elif isinstance(provided_types, str):
				types = self.device_types.append(provided_types)
			else:
				types = self.device_types
	
		devices = {}

		if types:
			for device_type in types:
				devices[device_type] = tf.config.experimental.list_physical_devices(device_type)

		if with_benchmark:
			benched = self.run_benchmarks()

		return devices

	def get_device_name(self, device):
		name = device.name.split(':')
		device_num = name[len(name)-1]
		device_name = device.device_type + ':' + str(device_num)
		return device_name

	def sort_devices(self, devices, sorter=None):
		sorted_devices = sorter(devices)
		return sorted_devices 

	def run_benchmarks(self):
		Bench = Benchmark.Benchmark(self.devices, self.config).benchmark_devices()