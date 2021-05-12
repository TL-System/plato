"""
A federated learning server using FedNova.

Reference:

Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated
Optimization" (https://arxiv.org/pdf/2007.07481.pdf)
"""
from collections import OrderedDict

from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedNova algorithm. """
    def federated_averaging(self, reports):
        """Aggregate weight updates from the clients using FedNova."""
        # Extracting updates from the reports
        updates = self.extract_client_updates(reports)

        # Extracting the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report, __) in reports])

        # Extracting the number of local epoches, tau_i, from the reports
        local_epochs = [report.epochs for (report, __) in reports]

        # Performing weighted averaging
        avg_update = avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in updates[0].items()
        }

        tau_eff = 0
        for i, update in enumerate(updates):
            report, __ = reports[i]
            num_samples = report.num_samples
            tau_eff_ = local_epochs[i] * num_samples / self.total_samples
            tau_eff += tau_eff_
            #tau_effs.append(tau_eff)

        for i, update in enumerate(updates):
            report, __ = reports[i]
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples
                                             ) * tau_eff / local_epochs[i]

        # Extract baseline model weights
        baseline_weights = self.trainer.extract_weights()

        # Load updated weights into model
        updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            updated_weights[name] = weight + avg_update[name]

        return updated_weights
