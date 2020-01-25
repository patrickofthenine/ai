import sys
import os
import argparse
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
import pprint

pp = pprint.PrettyPrinter(indent=4)
parser = argparse.ArgumentParser(description="Utility for object_detection")
parser.add_argument('--in_path', type=str, help='input path')

class PathUtil:
	def __init__(self):
		self.args = parser.parse_args()
		self.image_dirs = self.args.in_path
	
	def get_image_dirs(self):
		image_dirs = self.args.image_dirs
		dir_list = os.listdir(image_dirs)
		dirs = {}
		for dir in dir_list:
			if dir: 
				path = os.path.join(image_dirs, dir)
				dirs[dir] = path
		return dirs

	def get_xml_files_from_directory(self, directories):
		dir_xml_map = {}
		for directory, path in directories.items(): 
			xml_files = glob.glob(path + '/*.xml')
			dir_xml_map[directory] = xml_files
		return dir_xml_map

	def modify_xml_file(self, file):
		new_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		new_file_path = os.path.join(new_project_root, file)
		xml = ET.parse(new_file_path)

		#update file path
		path = xml.find('path')
		path.text = new_file_path

		#update object name
		root = xml.getroot()

		container_ship = []
		for box in root.iter('bndbox'):
			container = {}	
			for contents in box.iter():
				if contents.tag != 'bndbox':
					container[contents.tag] = contents.text
					container_ship.append(container)
			print(container)

		print(container_ship)
		print('...writing updates to file', file)
		xml.write(file)

		return

	def change_dir_path_in_xml(self):
		#Goes to supplied directory 
		#finds directories there.
		#finds .xml files in those directories
		#updates path in .xml file to match current xml file location

		directories = self.get_image_dirs()
		xml_files = self.get_xml_files_from_directory(directories)
		for directory, files in xml_files.items():
			for file in files:
				#print(file)
				try:
					self.modify_xml_file(file)
				except Exception as e:
					print(e)

PathUtil = PathUtil().change_dir_path_in_xml()