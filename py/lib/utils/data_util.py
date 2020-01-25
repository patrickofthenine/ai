import time
import os
import math
import shutil
from urllib.parse import urlparse
import requests
from xtract import xtract
from tqdm import tqdm

class DataUtil:
	def __init__(self, config, name):
		self.name = name 
		self.config = config
		self.extensions  = self._get_extension_list(['.rar', '.zip', '.tar'],['.gz', '.xz', '.bz2'])

	def get_and_expand(self):
		compressed = self._get_compressed()
		expanded = self._expand_compressed()

	def source_models(self):
		expanded = self.config[self.name]['expanded']
		
		for (walked) in os.walk(expanded):
			try:
				self._expand(walked, self._source_pipeline_configs)
			except exception as e: 
				print(e)

	def _build_extension(self, filename, ext=None):
		if not filename:
			return
		file, extension = os.path.splitext(filename)	
		if extension in self.extensions:
			extension = extension+ext if ext else extension	
			return self._build_extension(file, extension)
		else:
			return os.path.basename(filename), ext

	def _copy_file(self, source, destination):
		if not os.path.exists(destination):
			try:
				shutil.copyfile(source, destination)	
			except Exception as e:
				print(e)
		return 

	def _expand(self, walked, callback):
		expanded = self.config[self.name]['expanded']
		path, dirs, files = walked
		files = filter(lambda file: not file.startswith('.'), files)
		for file in files:
			source = os.path.join(path, file)
			relative = os.path.relpath(source, self.config[self.name]['compressed'])
			destination = os.path.join(expanded, relative)
			try:
				callback(source, destination)
			except Exception as e:
				print(e)

	def _expand_compressed(self):
		compressed = self.config[self.name]['compressed']	
		for (walked) in os.walk(compressed):
			self._expand(walked, self._extract)

	def _extract(self, source, destination):
		file = os.path.basename(source)
		name, extension = self._build_extension(file)
		target_dir = os.path.dirname(destination)
		tmp_dir = os.path.join(target_dir, 'tmp')
		tmp = os.path.join(tmp_dir, name)
		target = os.path.join(target_dir, name)
		if extension in self.extensions:
			if not os.path.exists(target):
				try: 
					if not os.path.exists(tmp_dir):
						os.makedirs(tmp_dir, exist_ok=True)
					if not os.path.exists(tmp):
						print('...expanding', source, tmp)
						xtract(source, destination=tmp, keep_intermediate=False, overwrite=True, all=True)
						
					extracted = os.path.join(tmp, name)
					
					if not os.path.exists(target_dir):	
						os.makedirs(target_dir, exist_ok=True)

					print('...moving')
					if os.path.exists(extracted):
						shutil.move(extracted, target)
					else:
						shutil.move(tmp, target)

					print('...finishing')
					shutil.rmtree(tmp_dir)
				except Exception as e: 
					print(e)
		else :
			if not os.path.exists(target_dir):
				os.makedirs(target_dir, exist_ok=True)
			return self._copy_file(source, target)
		return 

	def _fetch(self, obj, path, callback):
		for key, value in obj.items():
			if not isinstance(value, str):
				updated_path = os.path.join(path, key)
				self._fetch(value, updated_path, callback)

			#because of the for and not a return
			if isinstance(value, str):
				out = self._format_url(key, value, path)
				if not os.path.exists(out):
					container, ext = os.path.splitext(out)
					os.makedirs(path, exist_ok=True)
					callback(value, out)

	def _format_url(self, name, url, out):
		path = urlparse(url).path
		file = os.path.basename(path)
		file_name, ext = self._build_extension(file)
		out_path = os.path.join(out, name+ext) if ext else os.path.join(out, name)
		return out_path

	def _get_compressed(self):
		external = self.config[self.name]['external']	
		compressed = self.config[self.name]['compressed']

		self._fetch(external, compressed, self._req_and_write)

	def _get_extension_list(self, archives, compressions):
		valids =  archives + compressions

		for archive in archives:
			for compression in compressions:
				valids.append(archive+compression)
		return valids

	def _req(self, requirement, stream=True):
		print('...fetching ', requirement)
		r = requests.get(requirement, stream=stream)
		return r

	def _req_and_write(self, requirement, outpath):
		try: 
			r = self._req(requirement)
			r.raise_for_status()

			with open(outpath, 'wb') as out:
				self._write(r, out)
				r.close()	
		except Exception as e:
			print(e)

	def _source_pipeline_configs(self, file, destination=None):
		filename = os.path.basename(file)
		expanded = self.config[self.name]['expanded']
		relative = os.path.relpath(file, self.config[self.name]['expanded'])
		destination = os.path.join(self.config[self.name]['configs'], relative)
		if filename == 'pipeline.config':	
			try:
				if not os.path.exists(os.path.dirname(destination)):
					os.makedirs(os.path.dirname(destination))
				sourced = self._copy_file(file, destination)
				print('...pipeline.config copied', destination)
			except Exception as e:
				print(e)	
			return 	

	def _write(self, res, out):
		total = int(res.headers.get('content-length', 0))
		block = 8192
		counter = 0

		# tqdm lib provides progress bar
		for chunk in tqdm(res.iter_content(chunk_size=block), total=math.ceil(total//block), unit='B', unit_scale=True):
			if chunk:
				counter = counter + len(chunk)
				out.write(chunk)

		if (total != 0) and (counter != total):	
			print('Processed less data than expected', str(counter), str(total))
