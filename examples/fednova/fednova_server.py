"""
A federated learning server using FedNova.

Reference:

Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated
Optimization", in the Proceedings of NeurIPS 2020.

https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html
"""

from collections import OrderedDict

from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedNova algorithm. """
    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using FedNova."""
        # Extracting weights from the updates
        weights_received = self.extract_client_updates(updates)

        # Extracting the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report, __) in updates])

        # Extracting the number of local epoches, tau_i, from the updates
        local_epochs = [report.epochs for (report, __) in updates]

        # Performing weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        tau_eff = 0
        for i, update in enumerate(weights_received):
            report, __ = updates[i]
            num_samples = report.num_samples
            tau_eff_ = local_epochs[i] * num_samples / self.total_samples
            tau_eff += tau_eff_

        for i, update in enumerate(weights_received):
            report, __ = updates[i]
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples
                                             ) * tau_eff / local_epochs[i]

        return avg_update
