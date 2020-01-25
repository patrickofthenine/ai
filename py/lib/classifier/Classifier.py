import os
import tensorflow as tf
from py.lib.data.models import Modeler as Modeler
import pprint 

pp = pprint.PrettyPrinter(indent=2)

class Classifier:
	def __init__(self):
		self.name = 'Classifier'
		self.config = config

	def get_checkpoints_path(self, set_name):
		training_base = self.config['training']['checkpoints']
		checkpoints = os.path.join(training_base, 'by_dataset', set_name, 'small_set', 'checkpoints')
		return checkpoints

	def load_model_from_checkpoint(self, checkpoint_dir):
		model = Model.Model(self.config).base_model
		reweighted = model.load_weights(checkpoint_dir)
		pp.pprint(checkpoint_dir)
		pp.pprint(self.config)
		return

	def manual_setup(self):
		path_to_checkpoint = ''
		path_to_training_image = ''
		loaded_model = ''
		return

	def get_config(self, path):
		return 

	def get_label_index(self):
		return

	def get_test_images(self):
		return

	def run(self):
		checkpoints = self.get_checkpoints_path('NIST')
		model = self.load_model_from_checkpoint(checkpoints)


classifier = Classifier({})


'''
###ahura-mazda:models ps$ ../env/bin/python3 bone_classifier.py

from PIL import Image
import os
import tensorflow as tf
import np
print(os.getcwd())
from research.object_detection.utils import visualization_utils as vis_util
from research.object_detection.utils import label_map_util
from matplotlib import pyplot as plt

import argparse
import pprint
pp = pprint.PrettyPrinter(indent=4)
parser = argparse.ArgumentParser(description="Bone Classifier")
parser.add_argument('--test', type=str, help='specify to use comparison image directory instead of video file')

class BoneClassifier(object):
    def __init__(self):
        self.args = parser.parse_args()
        PATH_TO_MODEL = '/home/p/dev/object_detection/training/models/tuned_model/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
            self.sess = tf.Session(graph=self.detection_graph)


    def get_classification(self, img):
        # Bounding Box Detection.
        try:
            with self.detection_graph.as_default():
                category_index = self.labelify()
                # Expand dimension since the model expects image to have shape [1, None, None, 3].
                img_expanded = np.expand_dims(img, axis=0)
                (boxes, scores, classes, num) = self.sess.run(
                    [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                    feed_dict={self.image_tensor: img_expanded})

                vis_util.visualize_boxes_and_labels_on_image_array(
                    img,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=3)

                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                return img
        except Exception as e:
            print("Exception during classification", e)
        return

    def get_test_image_locations(self):
        test_image_path = '/home/p/dev/object_detection/training/images/comparison_images'
        images = os.listdir(test_image_path)
        image_paths = []
        for image in images:
            image = '/home/p/dev/object_detection/training/images/comparison_images/' + image
            print(image)
            image_paths.append(image)
        return image_paths

    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        desired_shape = (im_height, im_width, 3)
        np_array = np.array(image.getdata())
        try:
            reshaped = np_array.reshape(desired_shape)
            reshaped_retyped = reshaped.astype(np.uint8)
            return reshaped_retyped
        except Exception as e:
            print('Exception loading image into numpy array', e)

    def labelify(self):
        NUM_CLASSES = 90
        label_path = '/home/p/dev/object_detection/configs/label_map.pbtxt'
        label_map = label_map_util.load_labelmap(label_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def classify_frame(self, frame):
        image_array = self.load_image_into_numpy_array(frame)
        classification = self.get_classification(image_array)
        return classification 
            print('TEST!')
            image_paths = self.get_test_image_locations()
            counter = 0
            for image_path in image_paths:
                img = Image.open(image_path)
                img_np = self.load_image_into_numpy_array(img)
                classification = self.get_classification(img_np)
                if(counter%1==0):
                    try:
                        print(counter, 'opening', image_path)
                        img_from_array = Image.fromarray(classification)
                        img_from_array.show()

                    except Exception as e:
                        print('exception during plotting', e)
                counter = counter + 1
            print("Counter:", counter)


#Bone = BoneClassifier()
#Bone.classify_images()


'''