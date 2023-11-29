"""The server and client model of the ControlNet in split learning."""
# pylint:disable=import-error
from abc import abstractmethod
import os

import wget
import torch
from ControlNet.cldm.model import create_model, load_state_dict
from plato.config import Config


def get_node_name(name, parent_name):
    """Get the name of each component of the controlnet"""
    if len(name) <= len(parent_name):
        return False, ""
    p = name[: len(parent_name)]
    if p != parent_name:
        return False, ""
    return True, name[len(parent_name) :]


class ControlNetModel(torch.nn.Module):
    """
    The model class with initilization of ControlNet with pre-trained diffusion model.
    """

    def __init__(self, class_name) -> None:
        super().__init__()

        checkpoint_path = Config.params["checkpoint_path"]
        # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
        config_name = ".yaml"
        model = create_model(
            "examples/split_learning/controlnet_split_learning/ControlNetSplitLearning/cldm_v15_"
            + class_name
            + config_name
        ).cpu()
        control_net_model_path = os.path.join(checkpoint_path, "control_sd15_ini.ckpt")
        # get the controlnet initial model
        if not os.path.exists(control_net_model_path):
            # download the diffusion model
            diffusion_model_path = os.path.join(checkpoint_path, "v1-5-pruned.ckpt")
            if not os.path.exists(diffusion_model_path):
                wget.download(
                    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt",
                    diffusion_model_path,
                )
            pretrained_weights = torch.load(diffusion_model_path)
            if "state_dict" in pretrained_weights:
                pretrained_weights = pretrained_weights["state_dict"]

            scratch_dict = model.state_dict()

            target_dict = {}
            for k in scratch_dict.keys():
                is_control, name = get_node_name(k, "control_")
                if is_control:
                    copy_k = "model.diffusion_" + name
                else:
                    copy_k = k
                if copy_k in pretrained_weights:
                    target_dict[k] = pretrained_weights[copy_k].clone()
                else:
                    target_dict[k] = scratch_dict[k].clone()
            model.load_state_dict(target_dict, strict=True)
            torch.save(model.state_dict(), control_net_model_path)
        else:
            model.load_state_dict(
                load_state_dict(control_net_model_path, location="cpu")
            )
        model.learning_rate = 1e-5
        model.sd_locked = True
        model.only_mid_control = False
        self.model = model

    @abstractmethod
    def training_step(self, batch):
        """The training step"""

    @abstractmethod
    def validation_step(self, batch):
        """Validation step"""

    def forward(self, batch):
        "Forward function considering training and validation."
        if self.training:
            return self.training_step(batch)
        return self.validation_step(batch)


class ClientModel(ControlNetModel):
    """
    The client model.
    The inputs are the intermediate features.
    The outputs are the feature generated from conditions and prompts.
    """

    def __init__(self) -> None:
        super().__init__(class_name="client")

    def forward_to(self, batch):
        """
        The specific function for client model,
        forward to get the intermediate feature.
        """
        outputs = self.forward(batch)
        return outputs["control_output"]

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
    """
    The server model.
    During training, the inputs are the intermediate features.
    During testing, the inputs are conditions and prompts.
    """

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
