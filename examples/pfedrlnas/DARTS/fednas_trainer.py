from plato.trainers import basic
from plato.config import Config
import torch


class Trainer(basic.Trainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        if hasattr(Config().parameters.architect, "pretrain_path"):
            self.model.model.load_state_dict(
                torch.load(Config().parameters.architect.pretrain_path)
            )
