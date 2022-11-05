import logging

from plato.config import Config
from plato.servers import fedavg
from plato.utils import fonts

import pickle
import numpy as np


class Server(fedavg.Server):
    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.subnets_config = [None for i in range(Config().clients.total_clients)]
        self.neg_ratio = None

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        subnet_config = self.algorithm.sample_config(server_response)
        self.subnets_config[server_response["id"] - 1] = subnet_config
        server_response["subnet_config"] = subnet_config

        return server_response

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        client_id_list = [update.client_id for update in self.updates]
        self.neg_ratio = self.algorithm.nas_aggregation(
            self.subnets_config, weights_received, client_id_list
        )

    def weights_aggregated(self, updates):
        accuracy_list = [update.report.accuracy for update in updates]
        client_id_list = [update.client_id for update in self.updates]
        subnet_configs = []
        for client_id_ in client_id_list:
            client_id = client_id_ - 1
            subnet_config = self.subnets_config[client_id]
            subnet_configs.append(subnet_config)

        epoch_index = self.algorithm.model.extract_index(subnet_configs)
        self.algorithm.model.step(
            accuracy_list, self.neg_ratio, epoch_index, client_id_list
        )

        self.trainer.model = self.algorithm.model

    def server_will_close(self):
        flops = []
        for i in range(1, Config().clients.total_clients):
            cfg = self.subnets_config[i]
            if not cfg == None:
                logging.info("the config of client %s is %s", str(i), str(cfg))
                self.algorithm.model.model.set_active_subnet(
                    cfg["resolution"],
                    cfg["width"],
                    cfg["depth"],
                    cfg["kernel_size"],
                    cfg["expand_ratio"],
                )
                flops.append(self.algorithm.model.model.compute_active_subnet_flops())
        logging.info(
            fonts.colourize(
                f"[{self}] Average Flops of models is {np.mean(np.array(flops))}."
            )
        )
        save_config = f"{Config().server.model_path}/subnet_configs.pickle"
        with open(save_config, "wb") as f:
            pickle.dump((self.subnets_config, flops), f)
