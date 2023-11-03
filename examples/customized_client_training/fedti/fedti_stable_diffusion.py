"""
Implementation of Diffusion model.
"""
import os
import logging

import torch
from torch import nn
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor


# In Textual-Inversion we only train the newly added embedding vector,
# so lets freeze rest of the model parameters here
def freeze_params(params):
    for param in params:
        param.requires_grad = False


class Text2ImageSDPipeline(nn.Module):
    """A lightweight network."""

    def __init__(self, **kwargs):
        """Define the model."""
        super().__init__()
        # self.pretrained_model_name_or_path = "stabilityai/stable-diffusion-2"
        model_type, model_name = Text2ImageSDPipeline.get_pretrained_config_info(
            **kwargs
        )
        self.pretrained_model_name_or_path = f"{model_type}/{model_name}"
        # home_dir = os.path.expanduser("~")
        # catch_dir = ".cache/huggingface/hub/models--stabilityai--stable-diffusion-2"
        # self.pretrained_model_name_or_path = os.path.join(home_dir, catch_dir)

        # Create noise_scheduler for training
        self.noise_scheduler = DDPMScheduler.from_config(
            self.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="vae"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="unet"
        )

        self.weight_dtype = torch.float32

        # Freeze vae and unet
        freeze_params(self.vae.parameters())
        freeze_params(self.unet.parameters())

    def prepare_status(self, accelerator):
        """Preparing status of learning."""
        if accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae and unet to device
        self.vae.to(accelerator.device, dtype=self.weight_dtype)
        self.unet.to(accelerator.device, dtype=self.weight_dtype)

        # Keep vae in eval mode as we don't train it
        self.vae.eval()
        # Keep unet in train mode to enable gradient checkpointing
        self.unet.train()

    def forward(self, pixel_values, hidden_states):
        """Fowarding the vae and unet."""
        # Convert images to latent space
        latents = (
            self.vae.encode(pixel_values.to(dtype=self.weight_dtype))
            .latent_dist.sample()
            .detach()
        )
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        ).long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        noise_pred = self.unet(
            noisy_latents, timesteps, hidden_states.to(self.weight_dtype)
        ).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )
        return noise_pred, target

    def create_save_stable_diffusion_pipeline(
        self, prompt_learner, accelerator, output_dir, reload=False, save=True
    ):
        """Creating the pipeline and save it."""
        if accelerator.is_main_process:
            if (
                os.path.exists(output_dir)
                and "vae" in os.listdir(output_dir)
                and "unet" in os.listdir(output_dir)
                and reload
            ):
                logging.info("Loading a pre-trained pipeline from %s", output_dir)
                pipeline = StableDiffusionPipeline.from_pretrained(
                    output_dir,
                    # "downloaded_embedding",
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            else:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.pretrained_model_name_or_path,
                    text_encoder=accelerator.unwrap_model(prompt_learner).text_encoder,
                    vae=self.vae,
                    unet=self.unet,
                    tokenizer=prompt_learner.tokenizer,
                )
                # Recommended if your computer has < 64 GB of RAM
                pipeline.enable_attention_slicing()
                if save:
                    pipeline.save_pretrained(output_dir)

        return pipeline

    @staticmethod
    def get_pretrained_config_info(**kwargs):
        """Getting the pre-trained information of config."""
        # stabilityai
        model_type = kwargs["model_type"] if "model_type" in kwargs else "stabilityai"

        # for example, stable-diffusion-2
        model_name = (
            kwargs["model_name"] if "model_name" in kwargs else "stable-diffusion-2"
        )

        return model_type, model_name
