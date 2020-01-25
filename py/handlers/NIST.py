class NIST:
	def __init__(self, config):
		self.name = 'NIST'
		self.config = config

	def build_config(self):
		ignore_as_label = ['hsf_', 'train_']
		ignore_during_build = ['hsf_page']
		built = self.config['ignore_as_label'].extend(ignore_as_label)
		built = self.config['ignore_during_build'].extend(ignore_during_build)
		return self.config

	def get_config(self):
		config = self.build_config()
		return config

