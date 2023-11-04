"""Control Net on client"""
# pylint:disable=import-error
import torch
from ControlNet.cldm.cldm import ControlLDM, ControlNet, ControlledUnetModel
from ControlNet.ldm.modules.diffusionmodules.util import timestep_embedding
from ControlNet.ldm.util import default


# pylint:disable=no-member
# pylint:disable=invalid-name
# pylint:disable=too-few-public-methods
class OurControlledUnetModel(ControlledUnetModel):
    """Client Unet."""

    # pylint:disable=unused-argument
    def forward(self, x, timesteps=None, context=None, **kwargs):
        """Forward function."""
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


class OurControlLDM(ControlLDM):
    """On the client."""

    # pylint:disable=unused-argument
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """Apply the model forward."""
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
        return {
            "control_output": control,
            "sd_output": sd_output,
            "noise": noise,
            "timestep": t,
        }


def symsigmoid(x):
    "Symmetric sigmoid function $|x|*(2/sigma(x)-1)$"
    return torch.abs(x) * (2 * torch.nn.functional.sigmoid(x) - 1)


class OurControlNet(ControlNet):
    """Our controlnet on the client"""

    # pylint:disable=unused-argument
    def forward(self, x, hint, **kwargs):
        """Forward function."""
        h = hint + x.type(self.dtype)
        h = symsigmoid(h)
        # Here we need to quantizde fp16 and try it.
        h = h.half()
        return h
