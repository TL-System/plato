from plato.algorithms import fedavg
from plato.config import Config

import fedtools
from NASVIT.models.attentive_nas_dynamic_model import AttentiveNasDynamicModel


class ServerAlgorithm(fedavg.Algorithm):
    def __init__(self, trainer=None):
        super().__init__(trainer)
        self.current_subnet = None

    def extract_weights(self, model=None):
        payload = self.current_subnet.cpu().state_dict()
        return payload

    def load_weights(self, weights):
        pass

    def sample_config(self, server_response):
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
        self.current_subnet = subnet
        return subnet_config

    def nas_aggregation(self, subnets_config, weights_received, client_id_list):
        client_models = []
        subnet_configs = []
        for i, client_id_ in enumerate(client_id_list):
            client_id = client_id_ - 1
            subnet_config = subnets_config[client_id]
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
        return neg_ratio


class ClientAlgorithm(fedavg.Algorithm):
    def __init__(self, trainer=None):
        super().__init__(trainer)

    def extract_weights(self, model=None):
        if model is None:
            model = self.model
        return model.cpu().state_dict()

    def load_weights(self, weights):
        self.model.load_state_dict(weights, strict=True)

    def generate_client_model(self, subnet_config):
        return fedtools.sample_subnet_w_config(
            AttentiveNasDynamicModel(), subnet_config, False
        )
