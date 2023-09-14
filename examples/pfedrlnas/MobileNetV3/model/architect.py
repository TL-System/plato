"""
NAS architect in PerFedRLNAS, a wrapper over the supernet.
"""
import copy
import pickle
import sys
import os

import torch
from plato.config import Config

from .mobilenetv3_supernet import NasDynamicModel
from .config import get_config

sys.path.append("./examples/pfedrlnas/")
from VIT.nasvit_wrapper import architect


class Architect(architect.Architect):
    """The supernet wrapper, including supernet and arch parameters."""

    # pylint: disable=too-many-instance-attributes
    def initialization(self):
        """
        Initizalization function.
        """
        self.model = NasDynamicModel(supernet=get_config().supernet_config)
        if hasattr(Config().parameters.architect, "pretrain_path"):
            weight = torch.load(
                Config().parameters.architect.pretrain_path, map_location="cpu"
            )["model"]
            del weight["classifier.linear.linear.weight"]
            del weight["classifier.linear.linear.bias"]
            self.model.load_state_dict(weight, strict=False)

        self.client_nums = Config().clients.total_clients
        if (
            hasattr(Config().parameters.architect, " personalize_last")
            and Config().parameters.architect.personalize_last
        ):
            self.lasts = [
                copy.deepcopy(self.model.classifier) for _ in range(self.client_nums)
            ]
        self._initialize_alphas()
        self.optimizers = [
            torch.optim.Adam(
                alpha,
                lr=Config().parameters.architect.learning_rate,
                betas=(0.5, 0.999),
                weight_decay=Config().parameters.architect.weight_decay,
            )
            for alpha in self.arch_parameters()
        ]
        self.baseline = {}
        if Config().args.resume:
            save_config = f"{Config().server.model_path}/baselines.pickle"
            if os.path.exists(save_config):
                with open(save_config, "rb") as file:
                    self.baseline = pickle.load(file)
        self.lambda_time = Config().parameters.architect.lambda_time
        self.lambda_neg = Config().parameters.architect.lambda_neg
        self.device = Config().device()
