"""
This class uses torch.multiprocessing to launch multiple processes, each with its 
own Python interpreter and starts from the starting function provided.
"""

import logging
import torch

class Executor:
    """Launches multiple processes."""

    def __init__(self, process_num):
        self.reports = []
        self.pool = torch.multiprocessing.Pool(process_num)

    def prompt(self, report):
        """Callback function, called when the process finishes."""
        logging.info('Client #%s finished', report.client_id)
        if report:
            self.reports.append(report)

    def schedule(self, function, args):
        """Schedule a function to run in a separate process."""
        self.pool.apply_async(function, args=args, callback=self.prompt)

    def wait(self):
        """Wait for all processes in the pool to finish."""
        self.pool.close()
        self.pool.join()
