"""
A customized server for federated unlearning.

Federated unlearning allows clients to proactively erase their data from a trained model. The model
will be retrained from scratch during the unlearning process.

If the AdaHessian optimizer is used, it will reflect what the following paper proposed:

Liu et al., "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid
Retraining," in Proc. INFOCOM, 2022.

Reference: https://arxiv.org/abs/2203.07320
"""

import logging
import os

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated unlearning server that implements the federated unlearning baseline algorithm.

    When 'data_deletion_round' specified in the configuration, the server will enter a retraining
    phase after this round is reached, during which it will roll back to the minimum round number
    necessary for all the clients requesting data deletion.

    For example, if client #1 wishes to delete its data after round #2, the server first finishes
    its aggregation at round #2, then finds out whether or not client #1 was selected in one of the
    previous rounds. If it was, the server will roll back to the round when client #1 was selected
    for the first time, and starts retraining phases from there. Otherwise, it will keep training
    but with client #1 deleting a percentage of its data samples, according to `delete_data_ratio`
    in the configuration.
    """

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
        )

        self.retraining = False

        # A dictionary that maps client IDs to the first round when the server selected it
        self.round_first_selected = {}

    def clients_selected(self, selected_clients):
        """Remembers the first round that a particular client ID was selected."""
        for client_id in selected_clients:
            if client_id not in self.round_first_selected:
                self.round_first_selected[client_id] = self.current_round

    def training_will_start(self) -> None:
        """Additional tasks before selecting clients for the first round of training."""
        super().training_will_start()

        # Saving a checkpoint for round #0 before any training starts,
        # useful if we need to roll back to the very beginning, such as
        # in the federated unlearning process
        self.save_to_checkpoint()

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates while supporting retraining-aware filtering."""
        self.context.current_round = self.current_round
        original_updates = updates

        if not self.retraining:
            self.context.updates = updates
            avg_update = await self.aggregation_strategy.aggregate_deltas(
                updates, deltas_received, self.context
            )
            self.total_samples = sum(update.report.num_samples for update in updates)
            self.context.updates = original_updates
            return avg_update

        recent_mask = [update.staleness <= self.current_round for update in updates]
        recent_updates = [
            update for update, is_recent in zip(updates, recent_mask) if is_recent
        ]
        recent_deltas = [
            delta for delta, is_recent in zip(deltas_received, recent_mask) if is_recent
        ]

        if not recent_updates:
            # Fall back to the original set to avoid empty aggregation.
            self.context.updates = updates
            avg_update = await self.aggregation_strategy.aggregate_deltas(
                updates, deltas_received, self.context
            )
            self.total_samples = sum(update.report.num_samples for update in updates)
            self.context.updates = original_updates
            return avg_update

        self.context.updates = recent_updates
        avg_update = await self.aggregation_strategy.aggregate_deltas(
            recent_updates, recent_deltas, self.context
        )
        self.total_samples = sum(update.report.num_samples for update in recent_updates)
        self.context.updates = original_updates
        return avg_update

    def clients_processed(self) -> None:
        """Enters the retraining phase if a specific set of conditions are satisfied."""
        super().clients_processed()

        clients_to_delete = Config().clients.clients_requesting_deletion

        if (
            self.current_round == Config().clients.data_deletion_round
        ) and not self.retraining:
            # If data_deletion_round equals to the current round at server for the first time,
            # and the clients requesting retraining has been selected before, the retraining
            # phase starts
            earliest_round = self.current_round

            for client_id, first_round in self.round_first_selected.items():
                if client_id in clients_to_delete:
                    self.retraining = True

                    if earliest_round > first_round:
                        earliest_round = first_round

            if self.retraining:
                self.current_round = earliest_round - 1

                logging.info(
                    "[%s] Data deleted. Retraining from the states after round #%s.",
                    self,
                    self.current_round,
                )

                # Loading the saved model on the server for starting the retraining phase
                checkpoint_path = Config.params["checkpoint_path"]

                model_name = (
                    Config().trainer.model_name
                    if hasattr(Config().trainer, "model_name")
                    else "custom"
                )
                filename = f"checkpoint_{model_name}_{self.current_round}.pth"
                self.trainer.load_model(filename, checkpoint_path)

                logging.info(
                    "[Server #%d] Model used for the retraining phase loaded from %s.",
                    os.getpid(),
                    checkpoint_path,
                )

                if (
                    hasattr(Config().clients, "exact_retrain")
                    and Config().clients.exact_retrain
                ):
                    # Loading the PRNG states on the server in preparation for the retraining phase
                    logging.info(
                        "[Server #%d] Random states after round #%s restored for exact retraining.",
                        os.getpid(),
                        self.current_round,
                    )

                    self._restore_random_states(self.current_round, checkpoint_path)
