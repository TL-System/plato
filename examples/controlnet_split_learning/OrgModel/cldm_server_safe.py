"""Control Net on client"""
import torch

# pylint:disable=import-error
from ControlNet.cldm.cldm import ControlLDM, ControlNet, ControlledUnetModel
from ControlNet.ldm.modules.diffusionmodules.util import timestep_embedding


# pylint:disable=no-member
# pylint:disable=invalid-name
# pylint:disable=too-few-public-methods
class OurControlledUnetModel(ControlledUnetModel):
    """Client Unet."""

    # pylint:disable=unused-argument
    # pylint:disable
    def forward_train(
        self, sd_output, timesteps=None, context=None, control=None, **kwargs
    ):
        "Forward function"
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


class OurControlLDM(ControlLDM):
    """On the client."""

    def forward_train(self, control, sd_output, cond_txt, t):
        "Forward function"
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

    # pylint:disable=unused-argument
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """The inner function during forward."""
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond["c_crossattn"], 1)

        if cond["c_concat"] is None:
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=None,
                only_mid_control=self.only_mid_control,
            )
        else:
            hint = torch.cat(cond["c_concat"], 1)
            hint = 2 * hint - 1
            hint = self.first_stage_model.encode(hint)
            hint = self.get_first_stage_encoding(hint).detach()
            control_server_txt = torch.zeros((x_noisy.shape[0], 1, 768)).to(self.device)
            control = self.control_model(
                x=x_noisy,
                hint=hint,
                timesteps=t,
                context=control_server_txt,
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=control,
                only_mid_control=self.only_mid_control,
            )

        return eps


def symsigmoid(x):
    "Symmetric sigmoid function $|x|*(2/sigma(x)-1)$"
    return torch.abs(x) * (2 * torch.nn.functional.sigmoid(x) - 1)


class OurControlNet(ControlNet):
    """Our controlnet on the client"""

    # pylint:disable=unused-argument
    def forward_train(self, h, timesteps, context, **kwargs):
        """Forward function."""
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        outs = []

        h = h.to(torch.float32)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs

    def forward(self, x, hint, timesteps, context, **kwargs):
        "Forward function"
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        outs = []

        h = hint + x.type(self.dtype)
        h = symsigmoid(h)
        # Here we need to quantizde fp16 and try it.
        h = h.half()
        h = h.to(torch.float32)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs
