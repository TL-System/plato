"""
Customized Server for PerFedRLNAS.
"""

import logging
import os
import pickle
import sys
import time

import fedtools
import numpy as np

from plato.callbacks.server import ServerCallback
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.samplers import all_inclusive
from plato.servers import fedavg
from plato.utils import csv_processor


class FendasServerCallback(ServerCallback):
    "The PerFedRLNAS server callback"

    def on_clients_processed(self, server, **kwargs):
        if hasattr(Config().clients, "do_test") and Config().clients.do_test:
            # Updates the log for client test accuracies
            accuracy_csv_file = (
                f"{Config().params['result_path']}/{os.getpid()}_accuracy_more_info.csv"
            )
            for update in server.updates:
                round_time = (
                    update.report.training_time + update.report.comm_time
                    if not update.report.exceed
                    else 0
                )
                accuracy_row = [
                    server.current_round,
                    update.client_id,
                    update.report.accuracy,
                    round_time,
                    update.report.utilization,
                    update.report.budget,
                    server.model_size[update.client_id - 1],
                ]
                csv_processor.write_csv(accuracy_csv_file, accuracy_row)
            logging.info("[%s] All client reports have been processed.", server)

    def on_clients_selected(self, server, selected_clients, **kwargs):
        pass


class ServerSync(fedavg.Server):
    """The PerFedRLNAS server assigns and aggregates global model with different architectures."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        # pylint:disable=too-many-arguments
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        self.subnets_config = [None for i in range(Config().clients.total_clients)]
        self.neg_ratio = None
        self.process_begin = None
        self.process_end = None
        self.model_size = np.zeros(Config().clients.total_clients)

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        subnet_config = self.algorithm.sample_config(server_response)
        self.subnets_config[server_response["id"] - 1] = subnet_config
        server_response["subnet_config"] = subnet_config

        return server_response

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregates weights of models with different architectures."""
        self.process_begin = time.time()
        client_id_list = [update.client_id for update in self.updates]
        num_samples = [update.report.num_samples for update in self.updates]
        self.neg_ratio = self.algorithm.nas_aggregation(
            self.subnets_config, weights_received, client_id_list, num_samples
        )
        for payload, client_id in zip(weights_received, client_id_list):
            payload_size = sys.getsizeof(pickle.dumps(payload)) / 1024**2
            self.model_size[client_id - 1] = payload_size

    def weights_aggregated(self, updates):
        """After weight aggregation, update the architecture parameter alpha."""
        accuracy_list = [update.report.accuracy for update in updates]
        round_time_list = [
            update.report.training_time + update.report.comm_time
            for update in self.updates
        ]
        client_id_list = [update.client_id for update in self.updates]
        subnet_configs = []
        for client_id_ in client_id_list:
            client_id = client_id_ - 1
            subnet_config = self.subnets_config[client_id]
            subnet_configs.append(subnet_config)

        epoch_index = self.algorithm.model.extract_index(subnet_configs)
        self.algorithm.model.step(
            [accuracy_list, round_time_list, self.neg_ratio],
            epoch_index,
            client_id_list,
        )

        self.trainer.model = self.algorithm.model
        self.process_end = time.time()

    def server_will_close(self):
        for i in range(1, Config().clients.total_clients):
            cfg = self.subnets_config[i]
            if cfg:
                logging.info("the config of client %s is %s", str(i), str(cfg))

        # Use model_path if available, otherwise use default models/pretrained directory
        if hasattr(Config().server, "model_path"):
            model_dir = Config().server.model_path
        else:
            model_dir = "./models/pretrained"

        save_config = f"{model_dir}/subnet_configs.pickle"

        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        with open(save_config, "wb") as file:
            pickle.dump(self.subnets_config, file)

    def save_to_checkpoint(self) -> None:
        # Use model_path if available, otherwise use default models/pretrained directory
        if hasattr(Config().server, "model_path"):
            model_dir = Config().server.model_path
        else:
            model_dir = "./models/pretrained"

        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        save_config = f"{model_dir}/subnet_configs.pickle"
        with open(save_config, "wb") as file:
            pickle.dump(self.subnets_config, file)
        save_config = f"{model_dir}/baselines.pickle"
        with open(save_config, "wb") as file:
            pickle.dump(self.algorithm.model.baseline, file)
        return super().save_to_checkpoint()

    def get_logged_items(self) -> dict:
        logged_items = super().get_logged_items()
        acc_info = self.algorithm.get_baseline_accuracy_info()
        logged_items["clients_accuracy_mean"] = acc_info["mean"]
        logged_items["clients_accuracy_std"] = acc_info["std"]
        logged_items["clients_accuracy_max"] = acc_info["max"]
        logged_items["clients_accuracy_min"] = acc_info["min"]
        logged_items["server_overhead"] = self.process_end - self.process_begin
        logged_items["model_size"] = np.mean(self.model_size)
        return logged_items


# pylint:disable=too-many-instance-attributes
class ServerAsync(ServerSync):
    """The PerFedRLNAS server assigns and aggregates global model with different architectures."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
    ):
        # pylint:disable=too-many-arguments
        super().__init__(
            model, datasource, algorithm, trainer, callbacks=[FendasServerCallback]
        )
        self.subnets_config = [None for _ in range(Config().clients.total_clients)]
        self.neg_ratio = None
        self.process_begin = None
        self.process_end = None
        self.total_samples = 0
        self.model_size = np.zeros(Config().clients.total_clients)
        if self.datasource is None:
            self.datasource = datasource

    async def aggregate_weights(self, updates, baseline_weights, weights_received):  # pylint: disable=unused-argument
        """Aggregates weights of models with different architectures."""
        self.process_begin = time.time()
        client_id_list = [update.client_id for update in self.updates]
        num_samples = [update.report.num_samples for update in self.updates]
        self.total_samples = sum(num_samples)
        deltas = await self.compute_weight_deltas(weights_received, client_id_list)
        aggregation_weights = await self.calculate_aggregation_weight(updates, deltas)
        self.neg_ratio = self.algorithm.nas_aggregation_async(
            aggregation_weights, self.subnets_config, weights_received, client_id_list
        )
        for payload, client_id in zip(weights_received, client_id_list):
            payload_size = sys.getsizeof(pickle.dumps(payload)) / 1024**2
            self.model_size[client_id - 1] = payload_size

    async def compute_weight_deltas(self, weights_received, client_id_list):
        """The calculation of deltas in NAS is different."""
        baseline_weights = self.algorithm.model.model.state_dict()
        client_models = []
        subnet_configs = []
        for i, client_id_ in enumerate(client_id_list):
            client_id = client_id_ - 1
            subnet_config = self.subnets_config[client_id]
            client_model = fedtools.sample_subnet_w_config(
                self.algorithm.model.model, subnet_config, False
            )
            client_model.load_state_dict(weights_received[i], strict=True)
            client_models.append(client_model)
            subnet_configs.append(subnet_config)
        proxy_supernets = fedtools.generate_proxy_supernets(
            client_models, subnet_configs
        )
        proxy_supernets_weights = [
            proxy_supernet.state_dict() for proxy_supernet in proxy_supernets
        ]
        return self.algorithm.compute_weight_deltas(
            baseline_weights, proxy_supernets_weights
        )

    def weights_aggregated(self, updates):
        """After weight aggregation, update the architecture parameter alpha."""
        super().weights_aggregated(updates)
        # Save the current model for later retrieval when cosine similarity needs to be computed
        filename = f"model_{self.current_round}.pth"
        self.trainer.save_model(filename)

    # pylint:disable=attribute-defined-outside-init
    def configure(self) -> None:
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """
        super().configure()
        total_rounds = Config().trainer.rounds
        target_accuracy = None
        target_perplexity = None
        if hasattr(Config().trainer, "target_accuracy"):
            target_accuracy = Config().trainer.target_accuracy
        elif hasattr(Config().trainer, "target_perplexity"):
            target_perplexity = Config().trainer.target_perplexity
        if target_accuracy:
            logging.info(
                "Training: %s rounds or accuracy above %.1f%%\n",
                total_rounds,
                100 * target_accuracy,
            )
        elif target_perplexity:
            logging.info(
                "Training: %s rounds or perplexity below %.1f\n",
                total_rounds,
                target_perplexity,
            )
        else:
            logging.info("Training: %s rounds\n", total_rounds)
        self.init_trainer()
        # Prepares this server for processors that processes outbound and inbound
        # data payloads
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Server", server_id=os.getpid(), trainer=self.trainer
        )
        if not (hasattr(Config().server, "do_test") and not Config().server.do_test):
            if self.datasource is None and self.custom_datasource is None:
                self.datasource = datasources_registry.get(client_id=0)
            elif self.datasource is None and self.custom_datasource is not None:
                self.datasource = self.custom_datasource()
            self.testset = self.datasource.get_test_set()
            if hasattr(Config().data, "testset_size"):
                self.testset_sampler = all_inclusive.Sampler(
                    self.datasource, testing=True
                )
        # Initialize the test accuracy csv file if clients compute locally
        if hasattr(Config().clients, "do_test") and Config().clients.do_test:
            accuracy_csv_file = (
                f"{Config().params['result_path']}/{os.getpid()}_accuracy_more_info.csv"
            )
            accuracy_headers = [
                "round",
                "client_id",
                "accuracy",
                "round_time",
                "utilization",
                "budget",
                "model_size",
            ]
            csv_processor.initialize_csv(
                accuracy_csv_file, accuracy_headers, Config().params["result_path"]
            )

    async def calculate_aggregation_weight(self, updates, deltas_received):
        """Calculate aggregation weights according to the port."""
        # Constructing the aggregation weights to be used
        aggregation_weights = []

        for i, update in enumerate(deltas_received):
            report = updates[i].report
            staleness = updates[i].staleness
            num_samples = report.num_samples

            similarity = await self.cosine_similarity(update, staleness)
            staleness_factor = ServerAsync.staleness_function(staleness)

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

        return aggregation_weights

    async def cosine_similarity(self, update, staleness):
        """Compute the cosine similarity of the received updates and the difference
        between the current and a previous model according to client staleness."""
        # Loading the global model from a previous round according to staleness
        filename = f"model_{self.current_round - 2}.pth"
        model_path = Config().params["model_path"]
        model_path = f"{model_path}/{filename}"

        return fedtools.calculate_similarity(
            model_path, self.trainer.model, update, staleness
        )

    @staticmethod
    def staleness_function(staleness):
        """The staleness_function."""
        staleness_bound = 10

        if hasattr(Config().server, "staleness_bound"):
            staleness_bound = Config().server.staleness_bound

        staleness_factor = staleness_bound / (staleness + staleness_bound)

        return staleness_factor


if hasattr(Config().server, "synchronous") and not Config().server.synchronous:
    Server = ServerAsync
else:
    Server = ServerSync
