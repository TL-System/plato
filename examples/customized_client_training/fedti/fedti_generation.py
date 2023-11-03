"""
Implementation of text-to-image generation with stable diffusion.
"""

import os
from auxfl.models import stable_diffusion_gen

import torch
from plato.config import Config


def main():
    generation_config = {}

    n_prompt_images = Config().algorithm.generation.n_prompt_images
    n_steps = Config().algorithm.generation.n_steps

    base_path = os.path.dirname(Config().params["base_path"])
    experiments_dir = Config().algorithm.generation.experiments_dir
    experiments_dir = os.path.join(
        base_path,
        experiments_dir,
    )

    model_path = Config().algorithm.generation.model_path

    # "image_generation"
    generation_output_dir = Config().algorithm.generation.generation_output_dir

    # "learned_embeds.bin"
    model_name = Config().algorithm.generation.model_name

    # "grooty"
    concept_name = Config().algorithm.concept_name

    checkpoint_path = os.path.join(experiments_dir, model_path, model_name)

    learned_token = torch.load(checkpoint_path)
    placeholder_token = next(iter(learned_token))

    # if placeholder_token == "":
    #     placeholder_token = f"<{concept_name}>"
    #     learned_token[placeholder_token] = learned_token.pop("")
    #     torch.save(learned_token, checkpoint_path)

    folder_name = f"{concept_name}_{placeholder_token}"
    output_dir = os.path.join(experiments_dir, generation_output_dir, folder_name)

    base_context = Config().algorithm.generation.base_context
    prompts = stable_diffusion_gen.prepare_prompts(base_context, placeholder_token)
    stable_diffusion_gen.t2i_generation(
        prompts,
        learned_token,
        placeholder_token,
        generation_config={
            "n_prompt_images": n_prompt_images,
            "output_dir": output_dir,
            "n_steps": n_steps,
        },
    )


if __name__ == "__main__":
    main()
