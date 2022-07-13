"""
A basic personalized federated learning client
who performs the global learning and local learning.


At the current stage, the personalized client supports
follows properties:

1. saving train/test datasets statistics information
2. defining the personalized model. Only mlp models
    are included.
3. organizing a controllable train process.
    The personalization is performed only in the
    final round.

"""

import os
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
import copy

import torch

from plato.config import Config
from plato.clients import simple
from plato.models import general_mlps_register as general_MLP_model
from plato.utils.arrange_saving_name import get_format_name
from plato.utils.checkpoint_operator import perform_client_checkpoint_loading
from plato.clients import base
from plato.utils import fonts


@dataclass
class Report(base.Report):
    """Report from a simple client, to be sent to the federated learning server."""
    comm_time: float
    update_response: bool


class Client(simple.Client):
    """A basic personalized federated learning client."""

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None,
                 personalized_model=None):
        super().__init__(model, datasource, algorithm, trainer)

        # the personalized model here corresponds to the client's
        # personal needs.
        self.custom_personalized_model = personalized_model
        self.personalized_model = None

    def perform_data_statistics(self, dataset, dataset_sampler):
        """ Record the data statistics. """
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            sampler=dataset_sampler.get())
        data_labels = []
        for _, label in data_loader:
            data_labels.extend(label.tolist())
        num_samples = len(data_labels)
        labels_sample_count = Counter(data_labels)
        return labels_sample_count, num_samples

    def save_data_statistics(self):

        result_path = Config().params['result_path']

        save_location = os.path.join(result_path,
                                     "client_" + str(self.client_id))
        os.makedirs(save_location, exist_ok=True)
        filename = get_format_name(client_id=self.client_id,
                                   suffix="data_statistics",
                                   ext="json")

        save_file_path = os.path.join(save_location, filename)

        if not os.path.exists(save_file_path):

            train_data_sta, train_count = self.perform_data_statistics(
                self.trainset, self.sampler)
            test_data_sta, test_count = self.perform_data_statistics(
                self.testset, self.testset_sampler)

            with open(save_file_path, 'w') as fp:
                json.dump(
                    {
                        "train": train_data_sta,
                        "test": test_data_sta,
                        "train_size": train_count,
                        "test_size": test_count
                    }, fp)
            logging.info(f"Saved the {self.client_id}'s local data statistics")

    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
        super().load_data()

        if hasattr(Config().clients, "do_data_tranform_logging") and Config(
        ).clients.do_data_tranform_logging:
            self.save_data_statistics()

    def configure(self) -> None:
        """ Performing the general client's configure and then initialize the
            personalized model for the client. """
        super().configure()
        if self.custom_personalized_model is not None:
            self.personalized_model = self.custom_personalized_model
            self.custom_personalized_model = None

        if self.personalized_model is None:

            encode_dim = self.trainer.model.encode_dim
            model_name = Config().trainer.model_name
            personalized_model_name = Config().trainer.personalized_model_name
            if "mlp" in personalized_model_name:
                self.personalized_model = general_MLP_model.Model.get_model(
                    model_type=personalized_model_name, input_dim=encode_dim)
            elif personalized_model_name == model_name:
                self.personalized_model = copy.deepcopy(self.trainer.model)

            # logging the personalzied model's info
            file_name = get_format_name(client_id=self.client_id,
                                        model_name=personalized_model_name,
                                        suffix="personalized",
                                        ext="log")
            model_path = Config().params['model_path']
            to_save_dir = os.path.join(model_path,
                                       "client_" + str(self.client_id))
            os.makedirs(to_save_dir, exist_ok=True)

            to_save_path = os.path.join(to_save_dir, file_name)
            with open(to_save_path, 'w') as f:
                f.write(str(self.personalized_model))

            logging.info(
                "Saved the client%d's personalized model (%s) information to models/",
                self.client_id, personalized_model_name)

        # assign the client's personalized model to its trainer
        if self.trainer.personalized_model is None:
            self.trainer.set_client_personalized_model(self.personalized_model)

    def load_payload(self, server_payload) -> None:
        """Loading the server model onto this client.

            We need to address this case:
                There is a big model containing sub-modules A, B, and C
                We only utilize the module A as the global model.
                Then, in each round, only the parameter of A will be exchanged
                among the server and the client. This makes the client receive
                only the parameters of A from the server.
                As each client initializes the local model with random weights
                at each round and then relies on the received parameters to assign
                trained weights.
                Therefore, at the begining of local training, only the module A
                is initialized by the received parameters. However, the B and C
                is still the random weights.

            What we want:
                1.- If the received parameters contain all weights of the model,
                    these parameters will be assigned to the model direclty.
                2.- If the received parameters can only be assigned to one part
                    of the model, other parts' checkpoint from previous round
                    will be loaded and assigned to the model.
        """
        if self.algorithm.is_incomplete_weights(server_payload):
            logging.info(
                "[Client #%d] Received server payload with '%s', which is incompleted for the local model.",
                self.client_id,
                Config().trainer.global_model_name)

            filename, cpk_oper = perform_client_checkpoint_loading(
                client_id=self.client_id,
                model_name=Config().trainer.model_name,
                current_round=self.current_round - 1,
                run_id=None,
                epoch=Config().trainer.epochs,
                prefix=None,
                anchor_metric="round",
                mask_anchors=["epoch", "personalized"],
                use_latest=True)

            if filename is None:
                # using the client's local model that is randomly initialized
                # at this round
                pool_weights = copy.deepcopy(self.trainer.model).state_dict()
                logging.info(
                    "[Client #%d]. no checkpoint %s, complete server's payload with local initial model.",
                    self.client_id, filename)
            else:
                print("filename: ", filename)
                pool_weights = cpk_oper.load_checkpoint(filename)["model"]
                logging.info(
                    "[Client #%d]. Loaded latest round's checkpoint %s to complete",
                    self.client_id, filename)

            completed_payload = self.algorithm.complete_weights(
                server_payload, pool_weights=pool_weights)
        else:
            completed_payload = server_payload
        self.algorithm.load_weights(completed_payload)

    async def train(self):
        """The machine learning training workload on a client."""

        rounds = Config().trainer.rounds
        accuracy = -1
        # Perform model training
        if self.current_round < rounds and (
                not (hasattr(Config().clients, "only_personalization")
                     and Config().clients.only_personalization)):
            # if not (hasattr(Config().clients, "only_personalization")
            #         and Config().clients.only_personalization):
            logging.info(
                fonts.colourize(
                    f"[{self}] Started training in communication round #{self.current_round}."
                ))
            try:
                training_time = self.trainer.train(
                    self.trainset,
                    self.sampler,
                    current_round=self.current_round)
            except ValueError:
                await self.sio.disconnect()

        if (self.current_round == rounds
                and Config.clients.do_final_personalization) or (
                    hasattr(Config().clients, 'pers_learning_interval')
                    and self.current_round %
                    Config().clients.pers_learning_interval == 0):

            logging.info(
                fonts.colourize(
                    f"[{self}] Started personalized training in the communication round #{self.current_round}."
                ))
            try:
                training_time, accuracy = self.trainer.pers_train(
                    self.trainset,
                    self.sampler,
                    testset=self.testset,
                    testset_sampler=self.testset_sampler,
                    current_round=self.current_round)
            except ValueError:
                await self.sio.disconnect()

        elif (hasattr(Config().clients, 'do_test')
              and Config().clients.do_test) and (
                  not hasattr(Config().clients, 'test_interval')
                  or self.current_round % Config().clients.test_interval == 0):
            accuracy = self.trainer.test(self.testset, self.testset_sampler)
        else:
            accuracy = 0

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if accuracy == -1:
            # The testing process failed, disconnect from the server
            await self.sio.disconnect()

        # Do not print the accuracy if it is not computed
        if accuracy == 0:
            if hasattr(Config().trainer, 'target_perplexity'):
                logging.info("[%s] Test perplexity: %.2f", self, accuracy)
            else:
                logging.info("[%s] Test accuracy: %.2f%%", self,
                             100 * accuracy)

        comm_time = time.time()

        if hasattr(Config().clients,
                   'sleep_simulation') and Config().clients.sleep_simulation:
            sleep_seconds = Config().client_sleep_times[self.client_id - 1]
            avg_training_time = Config().clients.avg_training_time
            self.report = Report(self.sampler.trainset_size(), accuracy,
                                 (avg_training_time + sleep_seconds) *
                                 Config().trainer.epochs, comm_time, False)
        else:
            self.report = Report(self.sampler.trainset_size(), accuracy,
                                 training_time, comm_time, False)

        return self.report, weights
