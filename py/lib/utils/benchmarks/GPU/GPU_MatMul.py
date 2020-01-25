import tensorflow as tf
import time 

class benchmark:
	def start_benchmark(self, iterations, root_complexity):
		start = time.time()
		x = tf.random.uniform([root_complexity, root_complexity])

		for iteration in range(iterations):
			tf.matmul(x, x)		

		end = time.time()
		result = end - start
		print("benchmark_GPU_matmul:", result)
		return result