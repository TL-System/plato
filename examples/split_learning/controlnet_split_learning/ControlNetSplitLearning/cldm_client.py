"""ControlNet on client"""
from collections import defaultdict

# pylint:disable=import-error
import torch
from ControlNet.cldm.cldm import ControlLDM, ControlNet, ControlledUnetModel
from ControlNet.ldm.modules.diffusionmodules.util import timestep_embedding
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


# pylint:disable=no-member
# pylint:disable=invalid-name
# pylint:disable=too-few-public-methods
class ClientControlledUnetModel(ControlledUnetModel):
    """Our design of UNet on the client."""

    # pylint:disable=unused-argument
    def forward(self, x, timesteps=None, context=None, **kwargs):
        """
        Forward function of UNet of the server model.
        Inputs are latent, timesteps and prompts.
        """
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(
                timesteps, self.model_channels, repeat_only=False
            )
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            h = self.input_blocks[0](h, emb, context)

        hs.append(h)
        return hs


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
        control = self.control_model(
            x=x_noisy,
            hint=hint,
            timesteps=t,
            context=cond_txt,
        )
        diffusion_encoder_output = diffusion_model(
            x=x_noisy,
            timesteps=t,
            context=cond_txt,
        )

        return control, diffusion_encoder_output, cond_txt

    def p_losses(self, x_start, cond, t, noise=None):
        "Return p_losses."
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        control, sd_output, cond_txt = self.apply_model(x_noisy, t, cond)
        intermediate_features = IntermediateFeatures()
        intermediate_features["control_output"] = control
        intermediate_features["sd_output"] = sd_output
        intermediate_features["noise"] = noise
        intermediate_features["timestep"] = t
        intermediate_features["cond_txt"] = cond_txt
        return intermediate_features


class ClientControlNet(ControlNet):
    """Our design of control network on the client."""

    # pylint:disable=unused-argument
    def forward(self, x, hint, timesteps, context, **kwargs):
        """
        Forward function of control network in the client model.
        Inputs are processed latent, conditions,
            timsteps, and processed prompts.
        """
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        h = x.type(self.dtype)
        h = self.input_blocks[0](h, emb, context)
        h += guided_hint

        return h
