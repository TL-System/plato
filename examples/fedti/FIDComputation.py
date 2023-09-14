"""
For FID computation, we need to generate enough images from 
the models.
"""

import os
import glob

from plato.config import Config

from auxfl.evaluators.fid_evaluation import generate_images


def get_checkpoint_models(checkpoint_dir, model_pattern):
    """Getting models from the checkpoint."""

    # "checkpoint_generation_prompt_*.pth"
    file_pattern = model_pattern

    # Use glob to get a list of file paths that match the pattern
    file_paths = glob.glob(os.path.join(checkpoint_dir, file_pattern))
    file_paths = [file_path for file_path in file_paths if file_path.endswith(".pth")]

    # Extract the file name and file name without extension for each file
    file_names = [os.path.basename(file_path) for file_path in file_paths]
    file_names_no_ext = [
        os.path.splitext(os.path.basename(file_path))[0] for file_path in file_paths
    ]

    return file_paths, file_names_no_ext


def main():
    generation_config = {}

    n_prompt_images = Config().algorithm.generation.n_prompt_images
    n_steps = Config().algorithm.generation.n_steps
    "grooty"
    concept_name = Config().algorithm.concept_name
    base_context = Config().algorithm.generation.base_context

    base_path = os.path.dirname(Config().params["base_path"])
    experiments_dir = Config().algorithm.generation.experiments_dir
    experiments_dir = os.path.join(
        base_path,
        experiments_dir,
    )

    checkpoint_path = Config().algorithm.generation.model_path
    # "image_generation"
    generation_output_dir = Config().algorithm.generation.generation_output_dir

    # "learned_embeds.bin"
    model_pattern = Config().algorithm.generation.model_pattern

    file_paths, file_names = get_checkpoint_models(
        os.path.join(experiments_dir, checkpoint_path), model_pattern
    )

    for file_path, model_name in zip(file_paths, file_names):
        print(f"Loading {file_path} for evaluation.")

        checkpoint_file_path = file_path

        folder_name = f"{model_name}_{concept_name}"
        output_dir = os.path.join(experiments_dir, generation_output_dir, folder_name)

        generation_config = {
            "n_prompt_images": n_prompt_images,
            "output_dir": output_dir,
            "n_steps": n_steps,
        }

        generate_images(
            model_path=checkpoint_file_path,
            generate_config=generation_config,
            base_context=base_context,
        )


if __name__ == "__main__":
    main()
