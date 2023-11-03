"""
Implementation of generating images with stable diffusion.
"""
import os
from typing import Dict, List
import random

import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
import PIL

from auxfl.models import generation_prompt_model
from auxfl.models import stable_diffusion


def save_images(
    prompts: List[str], generated_images: List[PIL.Image.Image], output_dir: str
):
    # save images
    images_folder = os.path.join(output_dir, "generated_images")
    os.makedirs(images_folder, exist_ok=True)

    n_prompt_images = int(len(generated_images) / len(prompts))

    for idx, image in enumerate(generated_images):
        prompt = prompts[int(idx / n_prompt_images)]
        filename = f"{prompt}_{idx}.jpg"

        save_file_path = os.path.join(images_folder, filename)
        if os.path.exists(save_file_path):
            # Generate a unique ID
            unique_id = random.randint(1, 100000)
            filename = f"{prompt}_{idx}_{unique_id}.jpg"
        save_file_path = os.path.join(images_folder, filename)

        image.save(save_file_path)


def prepare_prompts(base_context, placeholder_token):
    if isinstance(base_context, str):
        basic_prompt = base_context.replace("{}", placeholder_token)
        basic_prompt = [basic_prompt]
    else:
        basic_prompt = [
            context.replace("{}", placeholder_token) for context in base_context
        ]
    return basic_prompt


def forward_generation(prompts, pipeline, generation_config):
    n_steps = generation_config["n_steps"]
    n_prompt_images = generation_config["n_prompt_images"]

    # generate images
    generated_images = pipeline(
        prompts,
        num_images_per_prompt=n_prompt_images,
        num_inference_steps=n_steps,
        guidance_scale=7.5,
    ).images

    return generated_images


def t2i_generation(
    prompts: List[str],
    learned_tokeng: Dict[str, torch.Tensor],
    placeholder_token: str,
    generation_config: dict,
    is_save_pretrained: bool = True,
):
    """Performing text-to-image generation by redefining a pipeline with pre-trained
    embedding."""
    output_dir = generation_config["output_dir"]

    if torch.cuda.is_available():
        accelerator = Accelerator()
    elif torch.backends.mps.is_available():
        # mps does not support the fp16 mixture precision
        accelerator = Accelerator(mixed_precision="no")
    else:
        accelerator = Accelerator(cpu=True)

    print("Device: ", accelerator.device)

    prompt_learner = generation_prompt_model.GenerationPromptLearner()

    prompt_learner.tokenizer_add_placeholder(placeholder_token)
    learned_tokeng[placeholder_token] = learned_tokeng[placeholder_token].to(
        accelerator.device
    )
    print("Before, placeholder_token: ", prompt_learner.placeholder_token)
    print("Before, placeholder_token_id: ", prompt_learner.placeholder_token_id)
    print(
        "Before, placeholder_token embed: ",
        prompt_learner.get_placeholder_embedding(),
    )
    print("learned placeholder_token embed: ", learned_tokeng[placeholder_token])
    prompt_learner.set_placeholder_embedding(learned_tokeng[placeholder_token])
    prompt_learner = accelerator.prepare(prompt_learner)
    prompt_learner.to(accelerator.device)
    print("After, placeholder_token: ", prompt_learner.placeholder_token)
    print("After, placeholder_token_id: ", prompt_learner.placeholder_token_id)
    print(
        "After, placeholder_token embed: ", prompt_learner.get_placeholder_embedding()
    )
    t2i_generator = stable_diffusion.Text2ImageSDPipeline()
    t2i_generator.prepare_status(accelerator)

    create_pipe_folder = os.path.join(output_dir, "pipeline")

    pipeline = t2i_generator.create_save_stable_diffusion_pipeline(
        prompt_learner=prompt_learner,
        accelerator=accelerator,
        output_dir=create_pipe_folder,
        save=is_save_pretrained,
    )
    pipeline = accelerator.prepare(pipeline)
    pipeline.to(accelerator.device)

    generated_images = forward_generation(prompts, pipeline, generation_config)
    save_images(prompts, generated_images, output_dir)


def t2i_generation_reload(prompts: List[str], generation_config: dict):
    """Performing text-to-image generation by reloding the saved pipeline."""
    output_dir = generation_config["output_dir"]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    pipeline = StableDiffusionPipeline.from_pretrained(
        output_dir,
        # "downloaded_embedding",
        torch_dtype=torch.float16,
        device_map="auto",
    ).to(device)

    # generating images
    generated_images = forward_generation(pipeline, generation_config)
    save_images(prompts, generated_images, output_dir)
