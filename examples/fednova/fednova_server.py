"""
A federated learning server using FedNova.

Reference:

Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated
Optimization", in the Proceedings of NeurIPS 2020.

https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html
"""

from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedNova algorithm."""

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using FedNova."""
        # Extracting the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Extracting the number of local epoches, tau_i, from the updates
        local_epochs = [update.report.epochs for update in updates]

        # Performing weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        tau_eff = 0
        for i, update in enumerate(deltas_received):
            num_samples = updates[i].report.num_samples
            tau_eff_ = local_epochs[i] * num_samples / self.total_samples
            tau_eff += tau_eff_

        for i, update in enumerate(deltas_received):
            num_samples = updates[i].report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += (
                    delta
                    * (num_samples / self.total_samples)
                    * tau_eff
                    / local_epochs[i]
                )

        return avg_update
