import os
import logging
from collections import OrderedDict
from torch import nn

os.environ['config_file'] = 'examples/configs/server.yml'
from plato.servers import fedavg

class fedReIdServer(fedavg.Server):
    def __init__(self, model=None, trainer=None):
        super().__init__(model, trainer)
        
    def extract_client_updates(self, reports):
        """Extract the model weight updates from a client's report."""
        # Extract weights from reports
        weights_received = [payload for (__, payload) in reports]
        return self.algorithm.compute_weight_updates(weights_received)

    def aggregate_weights(self, reports):
        """Aggregate the reported weight updates from the selected clients."""
        return self.federated_averaging(reports)

    def federated_averaging(self, reports):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract updates from the reports
        updates = self.extract_client_updates(reports)

        # Extract the total number of samples
        self.total_belive = sum(
            [report.belive for (report, __) in reports])
        
        if self.total_belive == 0.0:
            self.total_belive = 1.0
        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in updates[0].items()
        }

        for i, update in enumerate(updates):
            report, __ = reports[i]
            belive = report.belive
            logging.info("%d -> %f", i, belive)
            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (belive / self.total_belive)

        # Extract baseline model weights
        baseline_weights = self.algorithm.extract_weights()

        # Load updated weights into model
        updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            updated_weights[name] = weight + avg_update[name]

        return updated_weights

def main():
    """A Plato federated learning training session using a custom model. """
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    
    server = fedReIdServer(model=model)
    server.run()

if __name__ == "__main__":
    main()
    
