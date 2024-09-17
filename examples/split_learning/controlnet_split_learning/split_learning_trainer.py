"""Train the ControlNet model with split learning"""
import os
import logging
import numpy as np
import torch
import torchvision
from PIL import Image
from torchmetrics.image import fid
from torchmetrics.multimodal import clip_score
import einops
from pytorch_msssim import SSIM

from plato.config import Config
from plato.trainers import split_learning
from dataset import dataset_basic


class Trainer(split_learning.Trainer):
    """A split learning trainer to train ControlNet."""

    def process_samples_before_client_forwarding(self, examples):
        batch_size = examples["jpg"].size(0)
        return examples, batch_size

    def process_training_samples_before_retrieving(self, training_samples):
        inputs, targets = training_samples
        diffusion_inputs = dataset_basic.DiffusionInputs()
        for keys, values in inputs.items():
            diffusion_inputs[keys] = values
        return diffusion_inputs, targets

    # pylint:disable=unused-argument
    def server_forward_from(self, batch, config):
        examples, labels = batch
        # If we use privacy-preserving training, we do not need to send gradients back
        if not (
            hasattr(Config().parameters.model, "privacy_preserving")
            and Config().parameters.model.privacy_preserving
        ):
            control = torch.nn.Parameter(
                examples["control_output"],
                requires_grad=True,
            )
        else:
            control = examples["control_output"]
        timestep = examples["timestep"]
        outputs = self.model.model.forward_train(
            control,
            examples["sd_output"],
            examples["cond_txt"],
            timestep,
        )
        self.optimizer.zero_grad()
        loss = self.customize_loss_criterion(outputs, labels, timestep)
        loss.backward()
        if not (
            hasattr(Config().parameters.model, "privacy_preserving")
            and Config().parameters.model.privacy_preserving
        ):
            grad = control.grad
        else:
            grad = None

        return loss, grad, labels.size(0)

    def customize_loss_criterion(self, outputs, labels, timestep):
        "Customized loss criterion for diffusion model"
        loss = self.model.model.get_loss(outputs, labels, mean=False).mean(
            dim=[1, 2, 3]
        )
        loss_simple = loss.mean() * self.model.model.l_simple_weight
        loss_vlb = (self.model.model.lvlb_weights[timestep] * loss).mean()
        loss = loss_simple + self.model.model.original_elbo_weight * loss_vlb
        return loss

    def log_condition(self, img, path):
        "Log image during validation."
        grid = torchvision.utils.make_grid(img, nrow=4)
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.numpy()
        grid = np.clip(grid * 127.5 + 127.5, 0, 255).astype(np.uint8)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(grid).save(path)
        return img

    def forward_to_intermediate_feature(self, inputs, targets):
        self.model.model.to(self.device)
        inputs["jpg"] = inputs["jpg"].to(self.model.model.device)
        inputs["hint"] = inputs["hint"].to(self.model.model.device)
        with torch.no_grad():
            output_dict = self.model.training_step(inputs)

        output_dict["control_output"] = output_dict["control_output"].detach().cpu()
        for index, items in enumerate(output_dict["sd_output"]):
            output_dict["sd_output"][index] = items.detach().cpu()
        noise = output_dict["noise"].detach().cpu()
        output_dict["timestep"] = output_dict["timestep"].detach().cpu()
        output_dict["cond_txt"] = output_dict["cond_txt"].detach().cpu()
        output_dict.pop("noise")
        return output_dict, noise

    def update_weights_before_cut(self, current_weights, weights):
        """Update the weights before cut layer, called when testing accuracy."""
        # update the weights of client model
        for key, _ in weights.items():
            if "input_hint_block" in key or "input_blocks.0" in key:
                current_weights[key] = weights[key]
        return current_weights

    # test
    # test the validation mse
    # pylint: disable=unused-argument
    # pylint:disable=too-many-locals
    def test_model_split_learning(self, batch_size, testset, sampler=None):
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=sampler
        )
        torch.cuda.empty_cache()
        sim_cond_scores = 0
        metric_ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        fidscore = fid.FrechetInceptionDistance(feature=64)
        clipscore = clip_score.CLIPScore()
        self.model.to(self.device)
        self.model.model.to(self.device)
        log_dir = Config().params["result_path"]
        basic_dataset = dataset_basic.BasicDataset(task=Config.data.condition)
        for batch, _ in test_loader:
            with torch.no_grad():
                generate = self.model.model.log_images(batch)
                origin = batch["jpg"]
                hint = batch["hint"]
                origin = einops.rearrange(origin, "b h w c -> b c h w").clone()
                self.log_condition(
                    generate["samples_cfg_scale_9.00"].detach().cpu(),
                    log_dir + "samples.png",
                )
                self.log_condition(origin, log_dir + "org.png")
                generate = (
                    torch.clip(
                        generate["samples_cfg_scale_9.00"] * 127.5 + 127.5, 0, 255
                    )
                    .to(torch.uint8)
                    .detach()
                    .cpu()
                )
                origin = (
                    torch.clip(origin * 127.5 + 127 / 5, 0, 255)
                    .to(torch.uint8)
                    .detach()
                    .cpu()
                )
                fidscore.update(generate, real=False)
                fidscore.update(origin, real=True)
                clipscore.update(generate, batch["txt"])
                generate_conditions = []
                for index_img in range(generate.shape[0]):
                    generate_condition_img = generate[index_img]
                    generate_condition_img = einops.rearrange(
                        generate_condition_img, "c h w -> h w c"
                    )
                    generate_condition_img = generate_condition_img.numpy()
                    generate_condition_img = basic_dataset.process(
                        generate_condition_img
                    )
                    generate_conditions.append(generate_condition_img.tolist())
                generate_conditions = torch.tensor(
                    np.array(generate_conditions).astype(np.float32) / 255.0
                )
                sim_cond = metric_ssim(
                    einops.rearrange(generate_conditions, "b h w c-> b c h w"),
                    einops.rearrange(hint, "b h w c->b c h w"),
                ).item()
                sim_cond_scores += sim_cond
            sim_cond_scores /= len(test_loader)
        fidscores = fidscore.compute().detach().item()
        clipscores = clipscore.compute().detach().item() / 100
        logging.info(
            "[Server #%d] FID: %.4f, CLIP score: %.4f, condition MSE: %.4f",
            os.getpid(),
            fidscores,
            clipscores,
            sim_cond_scores,
        )
        torch.cuda.empty_cache()
        return fidscores / 100
