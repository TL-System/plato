"""
A federated learning server using FedADp to aggregate the training of Actor-Critic models.
"""

import math
import logging
import asyncio
from torch.autograd import Variable

from plato.servers import fedavg
from plato.config import Config
import pickle
import numpy as np
import csv
import random
import torch


class A2CServer(fedavg.Server):
    """Federated learning server using FedADP to agrregate Actor-Critic models."""

    """ A custom federated learning server. """

    def __init__(
        self, algorithm_name, env_name, model=None, trainer=None, algorithm=None
    ):
        super().__init__(trainer=trainer, algorithm=algorithm, model=model)
        self.algorithm_name = algorithm_name
        self.env_name = env_name

        self.actor_local_angles = {}
        self.critic_local_angles = {}
        self.last_global_actor_grads = None
        self.last_global_critic_grads = None
        self.actor_adaptive_weighting = None
        self.critic_adaptive_weighting = None
        self.global_actor_grads = None
        self.global_critic_grads = None

        logging.info("A custom server has been initialized.")

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging."""

        num_samples = [update.report.num_samples for update in updates]
        total_samples = sum(num_samples)

        self.global_actor_grads = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0][0].items()
        }

        self.global_critic_grads = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0][1].items()
        }

        # Perform weighted averaging for both Actor and Critic
        actor_avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0][0].items()
        }
        critic_avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0][1].items()
        }
        client_list = []
        client_path = f"{Config().results.results_dir}_seed_{Config().server.random_seed}/{Config().results.file_name}_client_saved"

        for i, update in enumerate(deltas_received):
            #_, report, _, _ = updates[i]
            update_from_actor, update_from_critic = update

            for name, delta in update_from_actor.items():
                self.global_actor_grads[name] += delta * (
                    num_samples[i] / total_samples
                )

            for name, delta in update_from_critic.items():
                self.global_critic_grads[name] += delta * (
                    num_samples[i] / total_samples
                )

        (
            self.actor_adaptive_weighting,
            self.critic_adaptive_weighting,
        ) = self.calc_adaptive_weighting(deltas_received, num_samples)

        # self.adaptive_weighting
        for i, update in enumerate(deltas_received):
            #__, report, __, __ = updates[i]
            client_id = updates[i].report.client_id

            update_from_actor, update_from_critic = update

            client_list.append(client_id)

            for name, delta in update_from_actor.items():
                actor_avg_update[name] += delta * self.actor_adaptive_weighting[i]

            for name, delta in update_from_critic.items():
                critic_avg_update[name] += delta * self.critic_adaptive_weighting[i]

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        # Save lists of clients
        if not Config().server.percentile_aggregate:
            if "adp" not in Config().results.results_dir:
                self.save_files(f'{client_path}{"_Fed_avg"}', client_list)
            else:
                self.save_files(f'{client_path}{"_Fed_ADP"}', client_list)
        else:
            self.save_files(f'{client_path}{"_percentile"}', client_list)

        return actor_avg_update, critic_avg_update

    def calc_adaptive_weighting(self, updates, num_samples):
        """Compute the weights for model aggregation considering both node contribution
        and data size."""
        # Get the node contribution
        actor_contribs, critic_contribs = self.calc_contribution(updates)

        actor_adap_weighting = [None] * len(updates)
        critic_adap_weighting = actor_adap_weighting

        total_actor_weight = 0.0
        total_critic_weight = total_actor_weight

        for i, actor_contrib in enumerate(actor_contribs):
            total_actor_weight += num_samples[i] * math.exp(actor_contrib)

        for i, critic_contrib in enumerate(critic_contribs):
            total_critic_weight += num_samples[i] * math.exp(critic_contrib)

        for i, actor_contrib in enumerate(actor_contribs):
            actor_adap_weighting[i] = (
                num_samples[i] * math.exp(actor_contrib)
            ) / total_actor_weight

        for i, critic_contrib in enumerate(critic_contribs):
            critic_adap_weighting[i] = (
                num_samples[i] * math.exp(critic_contrib)
            ) / total_critic_weight

        return actor_adap_weighting, critic_adap_weighting

    def calc_contribution(self, updates):
        """Calculate the node contribution based on the angle between the local
        and global gradients."""
        actor_angles, actor_contribs = [None] * len(updates), [None] * len(updates)
        critic_angles, critic_contribs = actor_angles, actor_contribs

        self.global_actor_grads, self.global_critic_grads = self.process_grad(
            self.global_actor_grads, self.global_critic_grads
        )

        for i, update in enumerate(updates):
            update_from_actor, update_from_critic = update
            local_actor_grads, local_critic_grads = self.process_grad(
                update_from_actor, update_from_critic
            )

            inner_actor = np.inner(self.global_actor_grads, local_actor_grads)
            inner_critic = np.inner(self.global_critic_grads, local_critic_grads)

            norms_actor = np.linalg.norm(self.global_actor_grads) * np.linalg.norm(
                local_actor_grads
            )
            norms_critic = np.linalg.norm(self.global_critic_grads) * np.linalg.norm(
                local_critic_grads
            )

            actor_angles[i] = np.arccos(np.clip(inner_actor / norms_actor, -1.0, 1.0))
            critic_angles[i] = np.arccos(
                np.clip(inner_critic / norms_critic, -1.0, 1.0)
            )

        for i, angle in enumerate(actor_angles):
            client_id = self.selected_clients[i]

            if client_id not in self.actor_local_angles.keys():
                self.actor_local_angles[client_id] = angle

            self.actor_local_angles[client_id] = (
                (self.current_round - 1) / self.current_round
            ) * self.actor_local_angles[client_id] + (1 / self.current_round) * angle

            # Non-linear mapping to node contribution
            alpha = (
                Config().algorithm.alpha if hasattr(Config().algorithm, "alpha") else 5
            )

            actor_contribs[i] = alpha * (
                1
                - math.exp(-math.exp(-alpha * (self.actor_local_angles[client_id] - 1)))
            )

        for i, angle in enumerate(critic_angles):
            client_id = self.selected_clients[i]

            if client_id not in self.critic_local_angles.keys():
                self.critic_local_angles[client_id] = angle

            self.critic_local_angles[client_id] = (
                (self.current_round - 1) / self.current_round
            ) * self.critic_local_angles[client_id] + (1 / self.current_round) * angle

            # Non-linear mapping to node contribution
            alpha = (
                Config().algorithm.alpha if hasattr(Config().algorithm, "alpha") else 5
            )

            critic_contribs[i] = alpha * (
                1
                - math.exp(
                    -math.exp(-alpha * (self.critic_local_angles[client_id] - 1))
                )
            )

        return actor_contribs, critic_contribs

    @staticmethod
    def process_grad(actor_grads, critic_grads):
        """Process by flattening the gradients"""
        actor_grads = list(
            dict(sorted(actor_grads.items(), key=lambda x: x[0].lower())).values()
        )
        critic_grads = list(
            dict(sorted(critic_grads.items(), key=lambda x: x[0].lower())).values()
        )

        actor_flattened = actor_grads[0]
        critic_flattend = critic_grads[0]

        for i in range(1, len(actor_grads)):
            actor_flattened = np.append(
                actor_flattened, -actor_grads[i] / Config().algorithm.learning_rate
            )

        for i in range(1, len(critic_grads)):
            critic_flattend = np.append(
                critic_flattend, -critic_grads[i] / Config().algorithm.learning_rate
            )

        return actor_flattened, critic_flattend

    def standardize_fisher(self, report):
        norm_fisher_actor, norm_fisher_critic = {}, {}

        for name in report.fisher_actor:
            min_actor = Variable(report.fisher_actor[name]).min()
            max_actor = Variable(report.fisher_actor[name]).max()
            norm_fisher_actor[name] = (report.fisher_actor[name] - min_actor) / (
                max_actor - min_actor
            )
            norm_fisher_actor[name][torch.isnan(norm_fisher_actor[name])] = 1.0
            mean_norm_actor = Variable(norm_fisher_actor[name]).mean()
            norm_fisher_actor[name] += 1 - mean_norm_actor

        for name in report.fisher_critic:
            min_critic = Variable(report.fisher_critic[name]).min()
            max_critic = Variable(report.fisher_critic[name]).max()
            norm_fisher_critic[name] = (report.fisher_critic[name] - min_critic) / (
                max_critic - min_critic
            )
            norm_fisher_critic[name][torch.isnan(norm_fisher_critic[name])] = 1.0
            mean_norm_critic = Variable(norm_fisher_critic[name]).mean()
            norm_fisher_critic[name] += 1 - mean_norm_critic

        return norm_fisher_actor, norm_fisher_critic

    def save_files(self, file_path, data):
        # To avoid appending to existing files, if the current round is one we write over
        with open(
            f"{file_path}.csv", "w" if self.current_round == 1 else "a"
        ) as filehandle:
            writer = csv.writer(filehandle)
            # writerow only takes iterables
            if type(data) is not list:
                writer.writerow([data])
            else:
                writer.writerow(data)

    def read_last_entry(self, file_path):
        """Reads the last entry of the file"""
        if self.current_round == 1:
            return None
        else:
            with open(f"{file_path}.csv", "r") as filehandle:
                rows = filehandle.readlines()
                last_entry = 0
                for row in rows:
                    last_entry = row
        return last_entry

    def select_metric(self, report):
        """Select which metric to use"""
        if Config().server.percentile_aggregate == "actor_loss":
            return report.actor_loss
        elif Config().server.percentile_aggregate == "critic_loss":
            return report.critic_loss
        elif Config().server.percentile_aggregate == "actor_grad":
            return report.actor_grad
        elif Config().server.percentile_aggregate == "critic_grad":
            return report.critic_grad
        elif Config().server.percentile_aggregate == "sum_actor_fisher":
            return report.sum_actor_fisher
        elif Config().server.percentile_aggregate == "sum_critic_fisher":
            return report.sum_critic_fisher

    def create_loss_lists(self, updates):
        """Creates the lost lists"""
        loss_list = []
        for update in updates:
            if Config().server.percentile_aggregate == "actor_loss":
                loss_list.append(update.report.actor_loss)
            elif Config().server.percentile_aggregate == "critic_loss":
                loss_list.append(update.report.critic_loss)
            elif Config().server.percentile_aggregate == "actor_grad":
                loss_list.append(update.report.actor_grad)
            elif Config().server.percentile_aggregate == "critic_grad":
                loss_list.append(update.report.critic_grad)
            elif Config().server.percentile_aggregate == "sum_actor_fisher":
                loss_list.append(update.report.sum_actor_fisher)
            elif Config().server.percentile_aggregate == "sum_critic_fisher":
                loss_list.append(update.report.sum_critic_fisher)
            # elif

        return loss_list

    def save_to_checkpoint(self):
        """Save a checkpoint for resuming the training session."""
        checkpoint_path = Config.params["checkpoint_path"]

        copy_algorithm = self.algorithm_name
        if "_" in copy_algorithm:
            copy_algorithm = copy_algorithm.replace("_", "")

        env_algorithm = f"{self.env_name}{copy_algorithm}"
        model_name = env_algorithm

        filename = f"{model_name}_{self.current_round}"
        logging.info(
            "[%s] Saving the checkpoint to %s/%s.", self, checkpoint_path, filename
        )
        self.trainer.save_model(filename, checkpoint_path)
        self.save_random_states(self.current_round, checkpoint_path)

        # Saving the current round in the server for resuming its session later on
        with open(
            f"{checkpoint_path}/current_round_seed_{Config().server.random_seed}.pkl",
            "wb",
        ) as checkpoint_file:
            pickle.dump(self.current_round, checkpoint_file)

    def save_random_states(self, round_to_save, checkpoint_path):
        """Saving the random states in the server for resuming its session later on."""
        states_to_save = [
            f"numpy_prng_state_{round_to_save}_seed_{Config().server.random_seed}",
            f"prng_state_{round_to_save}_seed_{Config().server.random_seed}",
        ]

        variables_to_save = [
            np.random.get_state(),
            random.getstate(),
        ]

        for i, state in enumerate(states_to_save):
            with open(f"{checkpoint_path}/{state}.pkl", "wb") as checkpoint_file:
                pickle.dump(variables_to_save[i], checkpoint_file)

    def restore_random_states(self, round_to_restore, checkpoint_path):
        """Restoring the numpy.random and random states from previously saved checkpoints
        for a particular round.
        """
        states_to_load = [
            f"numpy_prng_state_{round_to_restore}_seed_{Config().server.random_seed}",
            f"prng_state_{round_to_restore}_seed_{Config().server.random_seed}",
        ]
        variables_to_load = {}

        for i, state in enumerate(states_to_load):
            with open(
                f"{checkpoint_path}/{state}_{round_to_restore}.pkl", "rb"
            ) as checkpoint_file:
                variables_to_load[i] = pickle.load(checkpoint_file)

        numpy_prng_state = variables_to_load[0]
        self.prng_state = variables_to_load[1]

        np.random.set_state(numpy_prng_state)
        random.setstate(self.prng_state)

    def save_omega(self, client_id, omega_actor, omega_critic):
        omega_path = f"{Config().general.base_path}/{Config().server.model_path}"
        actor_path = f"{omega_path}/{self.env_name}{self.algorithm_name}omega_actor_client_{client_id}_seed_{Config().server.random_seed}.pth"
        critic_path = f"{omega_path}/{self.env_name}{self.algorithm_name}omega_critic_client_{client_id}_seed_{Config().server.random_seed}.pth"

        # read and aggregate omega if in round > 1
        if self.current_round > 1:
            with open(actor_path, "rb") as omg_actor_path:
                omega_actor_old = torch.load(omg_actor_path)
            with open(critic_path, "rb") as omg_critic_path:
                omega_critic_old = torch.load(omg_critic_path)

            for name, _ in omega_actor.items():
                omega_actor[name] = (
                    omega_actor[name] + (self.current_round - 1) * omega_actor_old[name]
                ) / self.current_round

            for name, _ in omega_critic.items():
                omega_critic[name] = (
                    omega_critic[name]
                    + (self.current_round - 1) * omega_critic_old[name]
                ) / self.current_round

        # Write omega in file
        with open(actor_path, "wb") as omg_actor_path:
            torch.save(omega_actor, omg_actor_path)
        with open(critic_path, "wb") as omg_critic_path:
            torch.save(omega_critic, omg_critic_path)
