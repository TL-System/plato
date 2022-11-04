from plato.clients import simple
from types import SimpleNamespace
import torch
from plato.config import Config
import torch.nn as nn
import fedtools
from NASVIT.models.attentive_nas_dynamic_model import AttentiveNasDynamicModel


class Client(simple.Client):
    """A personalized federated learning client using the FedRep algorithm."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

    def process_server_response(self, server_response) -> None:
        subnet_config = server_response["subnet_config"]
        self.algorithm.model = fedtools.sample_subnet_w_config(
            AttentiveNasDynamicModel(), subnet_config, False
        )
        self.trainer.model = self.algorithm.model
