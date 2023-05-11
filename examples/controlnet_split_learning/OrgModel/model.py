"""The server and client model of the original ControlNet."""
import torch
from ControlNet.cldm.model import create_model, load_state_dict
from plato.config import Config


class ClientModel(torch.nn.Module):
    """The client model."""

    def __init__(self) -> None:
        super().__init__()
        resume_path = Config().parameters.model.init_model_path
        learning_rate = 1e-5
        sd_locked = True
        only_mid_control = False

        # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
        model = create_model(Config().parameters.model.model_structure).cpu()
        model.load_state_dict(load_state_dict(resume_path, location="cpu"))
        model.learning_rate = learning_rate
        model.sd_locked = sd_locked
        model.only_mid_control = only_mid_control
        self.model = model

    # pylint:disable=invalid-name
    def training_step(self, batch):
        """The training step"""
        for k in self.model.ucg_training:
            p = self.model.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        output_dict = self.shared_step(batch)
        return output_dict

    @torch.no_grad
    def validation_step(self, batch):
        """Validation step"""
        output_dict = self.shared_step(batch)
        return output_dict

    def forward(self, batch):
        "Forward function"
        return self.training_step(batch)
