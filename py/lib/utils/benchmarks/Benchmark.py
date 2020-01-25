import os
import sys
import tensorflow as tf
import importlib

class Benchmark:
	def __init__(self, devices, config):
		self.devices = devices
		self.config = config

	def benchmark_devices(self):
		available_benchmarks = self.get_available_benchmarks()
		benchmark = self.run_benchmarks(self.devices, available_benchmarks, num_times=2)
		return

	def exec_benchmark(self, benchmark, num_times, iterations, root_complexity):
		benchmarked = []
		i = 0
		while(i < num_times):
			benchmarked.append(benchmark().start_benchmark(iterations, root_complexity))
			i = i + 1
		return benchmarked

	def filter_annoying(self, items):
		filter_chars = [".", "_"]
		filtered_items = []
		for item in items:
			for char in filter_chars:
				item = list(filter(lambda it: not it.startswith(char), item))
			filtered_items.append(item)
		return filtered_items

	def generate_benchmark(self, path):
		name, ext = os.path.splitext(os.path.basename(path))
		spec = importlib.util.spec_from_file_location(name, path)
		module = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(module)
		return module

	def get_available_benchmarks(self):
		benchmarks = {}
		for path, dirs, files in os.walk(self.config['benchmarks']):
			dirs, files = self.filter_annoying( (dirs, files) )	
			dirname = os.path.basename(path)

			if not dirname == 'benchmarks':
				benchmarks[dirname] = []
				for file in files:
					file_path = os.path.join(path,file)
					
					if file_path not in benchmarks[dirname]:
						benchmarks[dirname].append(file_path)

		return benchmarks

	def get_best_performing_device(self, results):
		best_performance = {
			'device': '',
			'time': 0,
		}

		for device, run_times in results.items():
			if len(run_times) > 0:
				total_run_time = 0
				for run_time in run_times:
					total_run_time = total_run_time + run_time  
				
				average_run_time = total_run_time / len(run_times) 
				print(device + ' average run time ' + str(average_run_time))
				if best_performance['time'] == 0:
					best_performance['device'] = device
					best_performance['time'] = average_run_time
				else:
					if best_performance['time'] > average_run_time:
						best_performance['device'] = device
						best_performance['time'] = average_run_time

		return best_performance

	def get_device_name(self, device):
		name = device.name.split(':')
		device_num = name[len(name)-1]
		device_name = device.device_type + ':' + str(device_num)
		return device_name

	def run_benchmarks(self, devices, benchmarks, num_times=3, iterations=1000, root_complexity=500):
		results = {}
		for device_type, device_list in devices.items():
			print("Num " + device_type + ': ' + str(len(device_list)))
			for device in device_list:
				dev = self.get_device_name(device)
				results[dev] = []
				with tf.device(dev):
					for benchmark_test_type, benchmark_paths in benchmarks.items():
						dev_type = dev.split(':')[0]
						if dev_type == benchmark_test_type or benchmark_test_type == 'shared':
							for benchmark_path in benchmark_paths:
								benchmark = self.generate_benchmark(benchmark_path).benchmark
								benchmark_results = self.exec_benchmark(benchmark, num_times, iterations, root_complexity)
								results[dev] = benchmark_results	

		performance = self.get_best_performing_device(results)
		print("Lowest average run time: ", performance['device'], performance['time'])
		return 

