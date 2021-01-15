"""
A simple federated learning server using federated averaging.
"""

import logging
import os

from servers import FedAvgServer
from config import Config


class AdaptiveSyncServer(FedAvgServer):
    """Federated averaging server with Adaptive Synchronization Frequency."""
    def __init__(self):
        super().__init__()
        self.previous_model = None

    async def process_reports(self):
        """Process the client reports by aggregating their weights."""
        updated_weights = self.aggregate_weights(self.reports)
        self.trainer.load_weights(updated_weights)

        # Testing the global model accuracy
        if Config().clients.do_test:
            # Compute the average accuracy from client reports
            self.accuracy = self.accuracy_averaging(self.reports)
            logging.info(
                '[Server {:d}] Average client accuracy: {:.2f}%.'.format(
                    os.getpid(), 100 * self.accuracy))
        else:
            # Test the updated model directly at the server
            self.accuracy = self.trainer.test(self.testset,
                                              Config().trainer.batch_size)
            logging.info('Global model accuracy: {:.2f}%\n'.format(
                100 * self.accuracy))

        await self.wrap_up_processing_reports()
