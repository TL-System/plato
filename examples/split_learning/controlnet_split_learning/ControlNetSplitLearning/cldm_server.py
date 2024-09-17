"""ControlNet on server."""
import torch

# pylint:disable=import-error
from ControlNet.cldm.cldm import ControlLDM, ControlNet, ControlledUnetModel
from ControlNet.ldm.modules.diffusionmodules.util import timestep_embedding


# pylint:disable=no-member
# pylint:disable=invalid-name
# pylint:disable=too-few-public-methods
class ServerControlledUnetModel(ControlledUnetModel):
    """Our design of UNet on the server."""

    # pylint:disable=unused-argument
    # pylint:disable
    def forward_train(
        self, sd_output, timesteps=None, context=None, control=None, **kwargs
    ):
        """
        Forward function of UNet of the server model.
        Inputs are processed outputs of the stable diffusion, timesteps,
            processed prompts and processed conditions.
        """
        with torch.no_grad():
            t_emb = timestep_embedding(
                timesteps, self.model_channels, repeat_only=False
            )
            emb = self.time_embed(t_emb)

            h = sd_output[0]
            for module in self.input_blocks[1:]:
                h = module(h, emb, context)
                sd_output.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for _, module in enumerate(self.output_blocks):
            if control is None:
                h = torch.cat([h, sd_output.pop()], dim=1)
            else:
                h = torch.cat([h, sd_output.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        return self.out(h)


class ServerControlLDM(ControlLDM):
    """Our design of ControlNet on the server."""

    def forward_train(self, control, sd_output, cond_txt, t):
        """
        Forward function of ControlNet in the server model.
        Inputs are processed conditions, outputs of stable diffusion and processed prompts.
        """
        diffusion_model = self.model.diffusion_model
        control = self.control_model.forward_train(
            h=control,
            timesteps=t,
            context=cond_txt,
        )
        eps = diffusion_model.forward_train(
            sd_output=sd_output,
            timesteps=t,
            context=cond_txt,
            control=control,
        )

        return eps


class ServerControlNet(ControlNet):
    """Our design of control network on the server."""

    # pylint:disable=unused-argument
    def forward_train(self, h, timesteps, context, **kwargs):
        """
        Forward function of control network in the server model.
        Inputs are processed conditions, timsteps, and processed prompts.
        """
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        outs = []

        outs.append(self.zero_convs[0](h, emb, context))
        for module, zero_conv in zip(self.input_blocks[1:], self.zero_convs[1:]):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs
