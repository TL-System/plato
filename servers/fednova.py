"""
A federated learning server using fednova.
This is a reimplementation of the following paper:
"Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization" (https://arxiv.org/pdf/2007.07481.pdf)
"""
import torch
from servers import FedAvgServer


class FedNovaServer(FedAvgServer):
    """Federated learning server using FedNova"""
    def federated_averaging(self, reports):
        """Aggregate weight updates from the clients using fednova."""
        # Extract updates from reports
        updates = self.extract_client_updates(reports)

        # Extract total number of samples
        self.total_samples = sum([report.num_samples for report in reports])

        # Extract local iteration tau_i from reports
        tau = self.extract_client_local_iteration(reports)

        # Perform weighted averaging
        avg_update = [torch.zeros(x.size()) for __, x in updates[0]]

        # compute tau_eff
        tau_eff = []
        for i, update in enumerate(updates):
            num_samples = reports[i].num_samples
            tau_eff_temp = 0

            for j, (___, delta) in enumerate(update):
                tau_eff_temp += tau[i] * num_samples / self.total_samples

            tau_eff.append(tau_eff_temp)

        # average and rescale updates with fednova
        for i, update in enumerate(updates):
            num_samples = reports[i].num_samples

            for j, (__, delta) in enumerate(update):
                avg_update[j] += delta * (
                    num_samples / self.total_samples) * tau_eff[i] / tau[i]

        # Extract baseline model weights
        baseline_weights = self.trainer.extract_weights()

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights

    def extract_client_local_iteration(self, reports):
        """Extract the model weight updates from a client's report."""
        # Extract local iteration from reports
        local_iteration_received = [report.iteration for report in reports]
        return local_iteration_received
