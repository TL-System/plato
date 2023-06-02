"""The server and client model of the original ControlNet."""
# pylint:disable=import-error
from abc import abstractmethod
import torch
from plato.config import Config
from ControlNet.cldm.model import create_model, load_state_dict


class ControlNetModel(torch.nn.Module):
    """The model class."""

    def __init__(self, class_name) -> None:
        super().__init__()
        resume_path = Config().parameters.model.init_model_path
        learning_rate = 1e-5
        sd_locked = True
        only_mid_control = False

        # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
        if not (
            hasattr(Config().parameters.model, "safe")
            and Config().parameters.model.safe
        ):
            config_name = ".yaml"
        else:
            config_name = "_safe.yaml"
        print(config_name)
        model = create_model(
            Config().parameters.model.model_structure + "_" + class_name + config_name
        ).cpu()
        model.load_state_dict(load_state_dict(resume_path, location="cpu"))
        model.learning_rate = learning_rate
        model.sd_locked = sd_locked
        model.only_mid_control = only_mid_control
        self.model = model

    @abstractmethod
    def training_step(self, batch):
        """The training step"""

    @abstractmethod
    def validation_step(self, batch):
        """Validation step"""

    def forward(self, batch):
        "Forward function"
        if self.training:
            return self.training_step(batch)
        return self.validation_step(batch)


class ClientModel(ControlNetModel):
    """The client model."""

    def __init__(self) -> None:
        super().__init__(class_name="client")

    # pylint:disable=invalid-name
    def training_step(self, batch):
        for k in self.model.ucg_training:
            p = self.model.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        output_dict = self.model.shared_step(batch)
        return output_dict

    @torch.no_grad()
    def validation_step(self, batch):
        output_dict = self.model.shared_step(batch)
        return output_dict


class ServerModel(ControlNetModel):
    """The server model."""

    def __init__(self) -> None:
        super().__init__(class_name="server")

    # pylint:disable=invalid-name
    def training_step(self, batch):
        output_dict = self.model.forward_train(
            batch["control_output"],
            batch["sd_output"],
            batch["cond_txt"],
            batch["timestep"],
        )
        return output_dict

    @torch.no_grad()
    def validation_step(self, batch):
        output_dict = self.model.test(batch)
        return output_dict
