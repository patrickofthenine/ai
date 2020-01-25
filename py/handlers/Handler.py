from .NIST import NIST as NIST
from .DefaultHandler import DefaultHandler as DefaultHandler
class Handler:
	def __init__(self):
		self.name = 'Handler'

	def get_base_handler_config(self):
		base = {
			'ignore_as_label': ['training'],
			'ignore_during_build': ['IGNORE_'],
		}
		return base

	def get_handler(self, handler):
		base_config = self.get_base_handler_config()
		if handler == 'NIST':
			return NIST(base_config).get_config()       	 
		else:
			return DefaultHandler(base_config)
