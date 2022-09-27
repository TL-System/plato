"""
An asynchronous federated learning client using Sirius.
"""
from types import SimpleNamespace
import numpy as np
from plato.clients import simple
from plato.config import Config

from inspect import currentframe

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

class Client(simple.Client):

	def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
		super().__init__(model, datasource, algorithm, trainer)
	
	def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
		"""Wrap up generating the report with any additional information."""
		print("Client finished line_", get_linenumber())
		train_squared_loss_steps = self.trainer.run_history.get_metric_values(
            "train_loss"
        )
		train_squared_loss_step = train_squared_loss_steps[-1]
		report.statistics_utility = report.num_samples * np.sqrt(
            1.0 / report.num_samples * train_squared_loss_step
        ) # zhifeng's original implementation return moving averaging norm of loss and calculate the stats utility on server side (clients_manager)
		print("Client finished line_", get_linenumber())
		return report
	
