import logging
from pickle import NONE

from plato.config import Config
from plato.servers import fedavg
from plato.utils import fonts

import pickle
import numpy as np
import random
import fedtools


class Server(fedavg.Server):
    """Federated learning server using federated averaging."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.subnets_config = [None for i in range(Config().clients.total_clients)]
        self.neg_ratio = None

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        if (
            hasattr(Config().parameters.architect, "max_net")
            and Config().parameters.architect.max_net
        ):
            subnet_config = self.trainer.model.model.sample_max_subnet()
        else:
            subnet_config = self.trainer.model.sample_config(
                client_id=server_response["id"] - 1
            )
        subnet = fedtools.sample_subnet_w_config(
            self.algorithm.model.model, subnet_config, True
        )
        self.subnets_config[server_response["id"] - 1] = subnet_config
        self.algorithm.current_subnet = subnet
        server_response["subnet_config"] = subnet_config

        return server_response

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        client_id_list = [update.client_id for update in self.updates]
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
        neg_ratio = fedtools.fuse_weight(
            self.algorithm.model.model,
            client_models,
            subnet_configs,
            [update.report.num_samples for update in self.updates],
        )
        self.neg_ratio = neg_ratio

    def weights_aggregated(self, updates):
        accuracy_list = [update.report.accuracy for update in updates]
        client_id_list = [update.client_id for update in self.updates]
        subnet_configs = []
        for client_id_ in client_id_list:
            client_id = client_id_ - 1
            subnet_config = self.subnets_config[client_id]
            subnet_configs.append(subnet_config)

        if (
            hasattr(Config().parameters.architect, "warmup")
            and self.current_round < Config().parameters.architect.warmup
        ):
            pass
        else:
            # update alpha i s with value net
            epoch_index = self.trainer.model.extract_index(subnet_configs)
            if (
                hasattr(Config().parameters.architect, "natural_policy")
                and Config().parameters.architect.natural_policy
            ):
                self.trainer.model.step(
                    accuracy_list, self.neg_ratio, epoch_index, client_id_list
                )
            else:
                self.trainer.model.step(
                    accuracy_list, self.neg_ratio, epoch_index, client_id_list
                )

        self.algorithm.model = self.trainer.model

    def server_will_close(self):
        flops = []
        for i in range(1, Config().clients.total_clients):
            cfg = self.subnets_config[i]
            if not cfg == None:
                logging.info("the config of client %s is %s", str(i), str(cfg))
                self.trainer.model.model.set_active_subnet(
                    cfg["resolution"],
                    cfg["width"],
                    cfg["depth"],
                    cfg["kernel_size"],
                    cfg["expand_ratio"],
                )
                flops.append(self.trainer.model.model.compute_active_subnet_flops())
        logging.info(
            fonts.colourize(
                f"[{self}] Average Flops of models is {np.mean(np.array(flops))}."
            )
        )
        save_config = f"{Config().server.model_path}/subnet_configs.pickle"
        with open(save_config, "wb") as f:
            pickle.dump((self.subnets_config, flops), f)
