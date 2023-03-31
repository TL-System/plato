"""
Customized Server for PerFedRLNAS.
"""
import os
import sys
import logging
import pickle
import time
import numpy as np


from plato.utils import csv_processor
from plato.config import Config
from plato.servers import fedavg
from plato.callbacks.server import ServerCallback
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.samplers import all_inclusive

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

class Server(fedavg.Server):
    """The PerFedRLNAS server assigns and aggregates global model with different architectures."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
    ):
        # pylint:disable=too-many-arguments
        super().__init__(model, datasource, algorithm, trainer, callbacks=[FendasServerCallback])
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

    async def aggregate_weights(
        self, updates, baseline_weights, weights_received
    ):  # pylint: disable=unused-argument
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
        save_config = f"{Config().server.model_path}/subnet_configs.pickle"
        with open(save_config, "wb") as file:
            pickle.dump(self.subnets_config, file)

    def save_to_checkpoint(self) -> None:
        save_config = f"{Config().server.model_path}/subnet_configs.pickle"
        with open(save_config, "wb") as file:
            pickle.dump(self.subnets_config, file)
        save_config = f"{Config().server.model_path}/baselines.pickle"
        with open(save_config, "wb") as file:
            pickle.dump(self.algorithm.model.baseline, file)
        return super().save_to_checkpoint()

    def get_logged_items(self) -> dict:
        logged_items = super().get_logged_items()
        logged_items["server_overhead"] = self.process_end - self.process_begin
        return logged_items

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
                "round_time",
                "utilization",
                "budget",
                "model_size",
            ]
            csv_processor.initialize_csv(
                accuracy_csv_file, accuracy_headers, Config().params["result_path"])