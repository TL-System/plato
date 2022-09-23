"""
A asynchronous federated learning client using Sirius.
"""

import torch

from plato.clients import simple
from plato.config import Config

class Client(simple.CLient):

	def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
		super().__init__(model, datasource, algorithm, trainer)
	
