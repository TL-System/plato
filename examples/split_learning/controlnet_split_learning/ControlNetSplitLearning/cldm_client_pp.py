"""Control Net on client"""
from collections import defaultdict

# pylint:disable=import-error
import torch
from ControlNet.cldm.cldm import ControlLDM, ControlNet
from ControlNet.ldm.util import default


class IntermediateFeatures(defaultdict):
    """
    A type class of intermediate features based on Python dictionary.
    """

    def to(self, device):
        """
        Convert the intermediate_features to the device.
        """
        self["control_output"] = self["control_output"].detach().to(device)
        self["cond_txt"] = self["cond_txt"].to(device)
        self["timestep"] = self["timestep"].to(device)
        sd_output = self["sd_output"]
        for index, items in enumerate(sd_output):
            self["sd_output"][index] = items.to(device)
        return self


class ClientControlLDM(ControlLDM):
    """Our design of ControlNet on the client."""

    # pylint:disable=unused-argument
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """
        Forward function of ControlNet in the client model.
        Inputs are noisy latents, timsteps, and prompts.
        """
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond["c_crossattn"], 1)
        hint = torch.cat(cond["c_concat"], 1)
        hint = 2 * hint - 1
        hint = self.first_stage_model.encode(hint)
        hint = self.get_first_stage_encoding(hint).detach()
        control = self.control_model(
            x=x_noisy,
            hint=hint,
        )
        diffusion_encoder_output = diffusion_model(
            x=x_noisy,
            timesteps=t,
            context=cond_txt,
        )

        return control, diffusion_encoder_output

    def p_losses(self, x_start, cond, t, noise=None):
        "Return p_losses."
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        control, sd_output = self.apply_model(x_noisy, t, cond)
        intermediate_features = IntermediateFeatures()
        intermediate_features["control_output"] = control
        intermediate_features["sd_output"] = sd_output
        intermediate_features["noise"] = noise
        intermediate_features["timestep"] = t
        return intermediate_features


def symsigmoid(x):
    "Symmetric sigmoid function $|x|*(2/sigma(x)-1)$"
    return torch.abs(x) * (2 * torch.nn.functional.sigmoid(x) - 1)


class ClientControlNet(ControlNet):
    """Our design of control network on the client."""

    # pylint:disable=unused-argument
    def forward(self, x, hint, timesteps, context, **kwargs):
        """
        Forward function of control network in the client model.
        Inputs are processed latent, conditions,
            timsteps, and processed prompts.
        """
        h = hint + x.type(self.dtype)
        h = symsigmoid(h)
        # Here we need to quantizde fp16 and try it.
        h = h.half()
        return h
