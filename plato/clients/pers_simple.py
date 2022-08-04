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
from plato.utils.checkpoint_operator import perform_client_checkpoint_loading, reset_all_weights
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

            logging.info(
                "Saved the client%d's personalized model (%s) information to models/",
                self.client_id, personalized_model_name)

        # assign the client's personalized model to its trainer
        if self.trainer.personalized_model is None:
            self.trainer.set_client_personalized_model(self.personalized_model)

        personalized_model_name = Config().trainer.personalized_model_name
        # logging the personalzied model's info
        file_name = get_format_name(client_id=self.client_id,
                                    model_name=personalized_model_name,
                                    suffix="personalized",
                                    ext="log")
        model_path = Config().params['model_path']
        to_save_dir = os.path.join(model_path, "client_" + str(self.client_id))
        os.makedirs(to_save_dir, exist_ok=True)

        to_save_path = os.path.join(to_save_dir, file_name)
        if not os.path.exists(to_save_path):
            with open(to_save_path, 'w') as f:
                f.write(str(self.personalized_model))

    def load_personalized_model(self):
        """ Initial the personalized model.

            There are three conditions to load the personalized model.
            1.- If it is required to maintain the personalized model,
                do_maintain_per_state: True
                if there is one existed personalized model
                    load the personalized model from previous round to initialize
                else
                    initialize the personalized model

        """

        # model_name = Config().trainer.model_name
        personalized_model_name = Config().trainer.personalized_model_name
        logging.info("[Client #%d] loading its personalized model [%s].",
                     self.client_id, personalized_model_name)

        filename, cpk_oper = perform_client_checkpoint_loading(
            client_id=self.client_id,
            model_name=personalized_model_name,
            current_round=self.current_round - 1,
            run_id=None,
            epoch=None,
            prefix="personalized",
            anchor_metric="round",
            mask_anchors=["epoch"],
            use_latest=True)

        if not (hasattr(Config().trainer, "do_maintain_per_state")
                and Config().trainer.do_maintain_per_state):
            # Do not use any previous personalized model, initial it from script
            logging.info(
                "[Client #%d] does not maintain personzalization status, thus initialize from script.",
                self.client_id)
            reset_all_weights(self.trainer.personalized_model)

        elif filename is None:
            # the personalized model has not been trained, the client needs to
            # start the training from script
            logging.info(
                "[Client #%d] has no trained personalized model (%s), thus initialize from script.",
                self.client_id, filename)
            reset_all_weights(self.trainer.personalized_model)
        else:
            logging.info(
                "[Client #%d] Loading latest round's trained personalized model %s.",
                self.client_id, filename)
            loaded_weights = cpk_oper.load_checkpoint(filename)
            self.trainer.personalized_model.load_state_dict(loaded_weights,
                                                            strict=True)
            logging.info(
                "[Client #%d] Completed the loading for initializing the personalized model.",
                self.client_id)

    def load_payload(self, server_payload) -> None:
        """Loading the server model onto this client.
            Loading the personalized model for current round.

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

            Then, if the client is activated and receives the payload from the server,
            it should load its personalized model for latter usage.

            What we want:
                1.- If no personalized model saved, the client will reinitialize its model.
                2.- If it has trained its personalized model, just load the latest one.

        """
        logging.info("[Client #%d] Received the global model: [%s].",
                     self.client_id,
                     Config().trainer.global_model_name)

        if self.algorithm.is_incomplete_weights(server_payload):
            logging.info(
                "[Client #%d] Received [%s], is incompleted to be assigned to the local model.",
                self.client_id,
                Config().trainer.global_model_name)

            filename, cpk_oper = perform_client_checkpoint_loading(
                client_id=self.client_id,
                model_name=Config().trainer.model_name,
                current_round=self.current_round - 1,
                run_id=None,
                epoch=None,
                prefix=None,
                anchor_metric="round",
                mask_anchors=["epoch", "personalized"],
                use_latest=True)

            if filename is None:
                # using the client's local model that is randomly initialized
                # at this round
                global_model_name = Config().trainer.global_model_name
                tmpl_model = copy.deepcopy(self.trainer.model)
                reset_all_weights(tmpl_model)
                pool_weights = tmpl_model.state_dict()
                logging.info(
                    "[Client #%d] no checkpoint %s, complete server's payload (the downloaded global model [%s]) with the local initial model.",
                    self.client_id, filename, global_model_name)
            else:
                pool_weights = cpk_oper.load_checkpoint(filename)["model"]
                logging.info(
                    "[Client #%d] Loaded latest round's checkpoint %s to complete",
                    self.client_id, filename)

            completed_payload, existed_prefixes, completed_prefixes = self.algorithm.complete_weights(
                server_payload, pool_weights=pool_weights)
            logging.info("[Client #%d] prefixes of the downloaded payload: %s",
                         self.client_id, ",".join(existed_prefixes))
            if completed_prefixes:
                logging.info("[Client #%d] prefixes used for completation: %s",
                             self.client_id, ",".join(completed_prefixes))
        else:
            completed_payload = server_payload

        self.algorithm.load_weights(completed_payload, strict=True)

        # Also, we load the personalization model.
        self.load_personalized_model()

    async def train(self):
        """The machine learning training workload on a client."""

        rounds = Config().trainer.rounds
        accuracy = -1
        training_time = 0.0
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

        elif (hasattr(Config().clients, 'do_test') and Config().clients.do_test
              ) and (hasattr(Config().clients, 'test_interval') and
                     self.current_round % Config().clients.test_interval == 0):
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
        if accuracy != 0:
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
