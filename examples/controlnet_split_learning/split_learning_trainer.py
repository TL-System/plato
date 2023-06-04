"""Train the ControlNet model with split learning"""
# pylint:disable=import-error
import os
import time
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
from split_learning import split_learning_trainer
from dataset.dataset_basic import process_condition


# pylint:disable=attribute-defined-outside-init
# pylint:disable=no-member
class Trainer(split_learning_trainer.Trainer):
    """The split learning algorithm to train ControlNet."""

    def _client_train_loop(self, examples):
        """Complete the client side training with gradients from server."""
        if not (
            hasattr(Config().parameters.model, "safe")
            and Config().parameters.model.safe
        ) and not (
            hasattr(Config().trainer, "jump_client") and Config().trainer.jump_client
        ):
            self.model.model = self.model.model.to(self.device)
            gradients = self.gradients[0].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model.forward(examples)

            # Back propagate with gradients from server
            outputs["control_output"].backward(gradients)
            self.optimizer.step()

        # No loss value on the client side
        loss = torch.zeros(1)
        self._loss_tracker.update(loss, examples["jpg"].size(0))
        return loss

    # pylint:disable=unused-argument
    def _server_train_loop(self, config, examples, labels):
        """The training loop on the server."""
        if not (
            hasattr(Config().trainer, "jump_server") and Config().trainer.jump_server
        ):
            self.model.model = self.model.model.to(self.device)
            if not (
                hasattr(Config().parameters.model, "safe")
                and Config().parameters.model.safe
            ):
                control = torch.nn.Parameter(
                    examples["control_output"].detach().to(self.model.model.device),
                    requires_grad=True,
                )
            else:
                control = (
                    examples["control_output"].detach().to(self.model.model.device)
                )

            if not (
                hasattr(Config().parameters.model, "safe")
                and Config().parameters.model.safe
            ):
                cond_txt = examples["cond_txt"].to(self.model.model.device)
            else:
                cond_txt = torch.zeros((control.shape[0], 1, 768)).to(
                    self.model.model.device
                )
            timestep = examples["timestep"].to(self.model.model.device)
            sd_output = examples["sd_output"]
            for index, items in enumerate(sd_output):
                sd_output[index] = items.to(self.model.model.device)
            outputs = self.model.model.forward_train(
                control,
                sd_output,
                cond_txt,
                timestep,
            )
            self.optimizer.zero_grad()
            loss = self.customize_loss_criterion(outputs, labels, timestep)
            loss.backward()
            loss = loss.cpu().detach()
            self._loss_tracker.update(loss, labels.size(0))
            # Record gradients within the cut layer
            if not (
                hasattr(Config().parameters.model, "safe")
                and Config().parameters.model.safe
            ):
                self.cut_layer_grad = [control.grad.cpu().clone().detach()]
            else:
                self.cut_layer_grad = [None]
            self.optimizer.step()

            logging.warning(
                "[Server #%d] Gradients computed with training loss: %.4f",
                os.getpid(),
                loss,
            )
        else:
            loss = torch.tensor([0])
            self._loss_tracker.update(loss, labels.size(0))
            if not (
                hasattr(Config().parameters.model, "safe")
                and Config().parameters.model.safe
            ):
                self.cut_layer_grad = [
                    examples["control_output"].detach().cpu().clone()
                ]
            else:
                self.cut_layer_grad = [None]

        return loss

    # test
    # test the validation mse
    # pylint: disable=unused-argument
    # pylint:disable=too-many-locals
    def test_model(self, config, testset, sampler=None, **kwargs):
        """
        Evaluates the model with the provided test dataset and test sampler.

        Auguments:
        testset: the test dataset.
        sampler: the test sampler. The default is None.
        kwargs (optional): Additional keyword arguments.
        """
        torch.cuda.empty_cache()
        batch_size = config["batch_size"]
        sim_cond_scores = 0
        metric_ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        fidscore = fid.FrechetInceptionDistance(feature=64)
        clipscore = clip_score.CLIPScore(
            model_name_or_path=os.path.join(
                Config().data.data_path,
                "controlnet/models/openai/clip-vit-large-patch14",
            )
        )

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=sampler
        )

        self.model.to(self.device)
        self.model.model.to(self.device)
        log_dir = Config().params["result_path"]
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
                    generate_condition_img = process_condition(
                        Config().data.condition, generate_condition_img
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

    # pylint: disable=unused-argument
    def train_model(self, config, trainset, sampler, **kwargs):
        """The default training loop when a custom training loop is not supplied."""
        torch.cuda.empty_cache()
        batch_size = config["batch_size"]
        self.sampler = sampler
        tic = time.perf_counter()

        self.run_history.reset()

        self.train_run_start(config)
        self.callback_handler.call_event("on_train_run_start", self, config)

        self.train_loader = self.get_train_loader(batch_size, trainset, sampler)

        # Initializing the loss criterion
        self._loss_criterion = self.get_loss_criterion()

        # Initializing the optimizer
        self.optimizer = self.get_optimizer(self.model)
        self.lr_scheduler = self.get_lr_scheduler(config, self.optimizer)
        self.optimizer = self._adjust_lr(config, self.lr_scheduler, self.optimizer)

        self.model.to(self.device)
        self.model.train()

        total_epochs = config["epochs"]

        self.model.to(self.device)
        self.model.model.to(self.device)
        for self.current_epoch in range(1, total_epochs + 1):
            self._loss_tracker.reset()
            self.train_epoch_start(config)
            self.callback_handler.call_event("on_train_epoch_start", self, config)

            for batch_id, (examples, labels) in enumerate(self.train_loader):
                self.train_step_start(config, batch=batch_id)
                self.callback_handler.call_event(
                    "on_train_step_start", self, config, batch=batch_id
                )

                labels = labels.to(self.device)

                loss = self.perform_forward_and_backward_passes(
                    config, examples, labels
                )

                self.train_step_end(config, batch=batch_id, loss=loss)
                self.callback_handler.call_event(
                    "on_train_step_end", self, config, batch=batch_id, loss=loss
                )

            self.lr_scheduler_step()

            if hasattr(self.optimizer, "params_state_update"):
                self.optimizer.params_state_update()

            # Simulate client's speed
            if (
                self.client_id != 0
                and hasattr(Config().clients, "speed_simulation")
                and Config().clients.speed_simulation
            ):
                self.simulate_sleep_time()

            # Saving the model at the end of this epoch to a file so that
            # it can later be retrieved to respond to server requests
            # in asynchronous mode when the wall clock time is simulated
            if (
                hasattr(Config().server, "request_update")
                and Config().server.request_update
            ):
                self.model.cpu()
                training_time = time.perf_counter() - tic
                filename = f"{self.client_id}_{self.current_epoch}_{training_time}.pth"
                self.save_model(filename)
                self.model.to(self.device)

            self.run_history.update_metric("train_loss", self._loss_tracker.average)
            self.train_epoch_end(config)
            self.callback_handler.call_event("on_train_epoch_end", self, config)

        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)
        torch.cuda.empty_cache()

    def customize_loss_criterion(self, outputs, labels, timestep):
        "Customied loss criterion for diffusion model"
        loss = self.model.model.get_loss(outputs, labels, mean=False).mean(
            dim=[1, 2, 3]
        )
        loss_simple = loss.mean() * self.model.model.l_simple_weight
        loss_vlb = (self.model.model.lvlb_weights[timestep] * loss).mean()
        loss = loss_simple + self.model.model.original_elbo_weight * loss_vlb
        return loss

    def log_condition(self, img, path):
        "log image"
        grid = torchvision.utils.make_grid(img, nrow=4)
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.numpy()
        grid = np.clip(grid * 127.5 + 127.5, 0, 255).astype(np.uint8)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(grid).save(path)
        return img
