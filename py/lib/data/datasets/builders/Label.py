class Label:
    def __init__(self):
        print('label class')
        
	def get_file_label_index(self, class_dirs_path, labels_index):
		class_dirs = [os.path.join(class_dirs_path, class_dir_name) for class_dir_name in os.listdir(class_dirs_path)]
		return True	

	def get_label_dataset(self, labels):
		label_dataset = tf.data.Dataset.from_tensor_slices(labels)
		return label_dataset

	def get_label_from_dirname(self, dirname):
		label = os.path.basename(dirname)
		return label