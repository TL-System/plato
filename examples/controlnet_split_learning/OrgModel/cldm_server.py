"""Control Net on client"""
import torch
from ControlNet.cldm.cldm import ControlLDM, ControlNet, ControlledUnetModel
from ControlNet.ldm.modules.diffusionmodules.util import timestep_embedding
from ControlNet.ldm.util import default


# pylint:disable=invalid-name
# pylint:disable=too-few-public-methods
class OurControlledUnetModel(ControlledUnetModel):
    """Client Unet."""

    # pylint:disable=unused-argument
    def forward(
        self,
        sd_output,
        timesteps=None,
        context=None,
        control=None,
        only_mid_control=False,
        **kwargs
    ):
        "Forward function"
        with torch.no_grad():
            t_emb = timestep_embedding(
                timesteps, self.model_channels, repeat_only=False
            )
            emb = self.time_embed(t_emb)

        h = sd_output.pop()
        if control is not None:
            h += control.pop()

        for _, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, sd_output.pop()], dim=1)
            else:
                h = torch.cat([h, sd_output.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        return self.out(h)


class OurControlLDM(ControlLDM):
    """On the client."""

    def forward(self, control, sd_output, cond_txt, t):
        "Forward function"
        diffusion_model = self.model.diffusion_model
        control = self.control_model(
            h=control,
            timesteps=t,
            context=cond_txt,
        )
        eps = diffusion_model(
            sd_output=sd_output,
            timesteps=t,
            context=cond_txt,
        )

        return eps


class OurControlNet(ControlNet):
    """Our controlnet on the client"""

    # pylint:disable=unused-argument
    def forward(self, h, timesteps, context, **kwargs):
        """Forward function."""
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        outs = []

        outs.append(self.zero_convs[0](h, emb, context))
        for module, zero_conv in zip(self.input_blocks[1:], self.zero_convs[1:]):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return h
