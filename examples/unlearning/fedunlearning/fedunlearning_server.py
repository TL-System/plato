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

import torch
from lib_mia import mia_server

from plato.config import Config


class Server(mia_server.Server):
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

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        self.retraining = False

        # A dictionary that maps client IDs to the first round when the server selected it
        self.round_first_selected = {}
        # A dictionary that maps client IDs to their sample indices
        self.sample_indices = {}

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
        """Aggregate the reported weight updates from the selected clients.
        If in retraing phase, staleness clients should not be aggregated.
        Otherwise the stale clients updates contains old model's info,
        will be aggregated after data_deletion_round.
        """
        if not self.retraining:
            self.context.updates = updates
            self.context.current_round = self.current_round
            deltas = await self.aggregation_strategy.aggregate_deltas(
                updates, deltas_received, self.context
            )
            self.total_samples = sum(update.report.num_samples for update in updates)
            return deltas

        recent_mask = list(
            map(lambda update: update.staleness <= self.current_round, updates)
        )
        recent_updates = list(
            filter(lambda update: update.staleness <= self.current_round, updates)
        )

        recent_deltas_received = [
            delta for delta, fresh in zip(deltas_received, recent_mask) if fresh
        ]

        if not recent_updates:
            if deltas_received:
                zero_delta = {
                    name: tensor.clone().zero_()
                    for name, tensor in deltas_received[0].items()
                }
            else:
                baseline = self.algorithm.extract_weights()
                zero_delta = {
                    name: weight.clone().zero_() for name, weight in baseline.items()
                }
            return zero_delta

        self.context.updates = recent_updates
        self.context.current_round = self.current_round

        deltas = await self.aggregation_strategy.aggregate_deltas(
            recent_updates, recent_deltas_received, self.context
        )
        self.total_samples = sum(update.report.num_samples for update in recent_updates)

        return deltas

    def clients_processed(self):
        """Enters the retraining phase if a specific set of conditions are satisfied."""
        super().clients_processed()

        # MIA evaluation after unlearning
        if (
            hasattr(Config().server, "mia_eval")
            and Config().server.mia_eval
            and self.current_round == Config().server.mia_eval_round
            and self.retraining
        ):
            self._perform_mia()

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
