"""
A federated learning server using Port.

Reference:

"How Asynchronous can Federated Learning Be?"

"""

import asyncio
import copy
import logging
import os

import torch
import torch.nn.functional as F

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedAsync algorithm."""

    async def cosine_similarity(self, update, staleness):
        """Compute the cosine similarity of the received updates and the difference
        between the current and a previous model according to client staleness."""
        # Loading the global model from a previous round according to staleness
        filename = f"model_{self.current_round - 2}.pth"
        model_path = Config().params["model_path"]
        model_path = f"{model_path}/{filename}"

        similarity = 1.0

        if staleness > 1 and os.path.exists(model_path):
            previous_model = copy.deepcopy(self.trainer.model)
            previous_model.load_state_dict(torch.load(model_path))

            previous = torch.zeros(0)
            for __, weight in previous_model.cpu().state_dict().items():
                previous = torch.cat((previous, weight.view(-1)))

            current = torch.zeros(0)
            for __, weight in self.trainer.model.cpu().state_dict().items():
                current = torch.cat((current, weight.view(-1)))

            deltas = torch.zeros(0)
            for __, delta in update.items():
                deltas = torch.cat((deltas, delta.view(-1)))

            similarity = F.cosine_similarity(current - previous, deltas, dim=0)

        return similarity

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Constructing the aggregation weights to be used
        aggregation_weights = []

        for i, update in enumerate(deltas_received):
            report = updates[i].report
            staleness = updates[i].staleness
            num_samples = report.num_samples

            similarity = await self.cosine_similarity(update, staleness)
            staleness_factor = Server.staleness_function(staleness)

            similarity_weight = (
                Config().server.similarity_weight
                if hasattr(Config().server, "similarity_weight")
                else 1
            )
            staleness_weight = (
                Config().server.staleness_weight
                if hasattr(Config().server, "staleness_weight")
                else 1
            )

            logging.info("[Client %s] similarity: %s", i, (similarity + 1) / 2)
            logging.info(
                "[Client %s] staleness: %s, staleness factor: %s",
                i,
                staleness,
                staleness_factor,
            )
            raw_weight = (
                num_samples
                / self.total_samples
                * (
                    (similarity + 1) / 2 * similarity_weight
                    + staleness_factor * staleness_weight
                )
            )
            logging.info("[Client %s] raw weight = %s", i, raw_weight)

            aggregation_weights.append(raw_weight)

        # Normalize so that the sum of aggregation weights equals 1
        aggregation_weights = [
            i / sum(aggregation_weights) for i in aggregation_weights
        ]

        logging.info(
            "[Server #%s] normalized aggregation weights: %s",
            os.getpid(),
            aggregation_weights,
        )

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        for i, update in enumerate(deltas_received):
            for name, delta in update.items():
                avg_update[name] += delta * aggregation_weights[i]

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update

    def weights_aggregated(self, updates):
        """
        Method called at the end of aggregating received weights.
        """
        # Save the current model for later retrieval when cosine similarity needs to be computed
        filename = f"model_{self.current_round}.pth"
        self.trainer.save_model(filename)

    @staticmethod
    def staleness_function(staleness):
        """The staleness_function."""
        staleness_bound = 10

        if hasattr(Config().server, "staleness_bound"):
            staleness_bound = Config().server.staleness_bound

        staleness_factor = staleness_bound / (staleness + staleness_bound)

        return staleness_factor
