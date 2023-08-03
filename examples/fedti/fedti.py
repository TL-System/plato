"""
Implementation of Textual Inversion [1] under the federated learning.

[1]. Gal, Rinon, et.al, An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion, ICLR23.

Textual inversion is a technique that allows us to add new styles or objects 
to text-to-image models without modifying the underlying model.


"""


from plato.algorithms import fedavg
from auxfl.clients import fed_generation_prompt_client
from auxfl.trainers import fed_t2i_prompt_trainer
from auxfl.client_callbacks import prompt_client_callbacks

from auxfl.datasources import oxford_tpet
from auxfl.datasources import fgvc_aircraft
from auxfl.datasources import flowers102
from auxfl.datasources import celeba

from plato.config import Config

import generation_model
import fedti_stable_diffusion
import fedti_client
import fedti_trainer


datasets = {
    "OxfordIIITPetOneClass": oxford_tpet,
    "FGVCAircraftOneClass": fgvc_aircraft,
    "Flowers102OneClass": flowers102,
    "CelebAOneClass": celeba,
}


def main():
    """
    A personalized federated learning sesstion for generation approach.
    """
    data_name = Config().data.datasource
    target_dataset = datasets[data_name]

    trainer = fed_t2i_prompt_trainer.Trainer

    # define the prompt learner for t2i generation
    prompt_learner = generation_prompt_model.GenerationPromptLearner

    client = fed_generation_prompt_client.Client(
        model=prompt_learner,
        trainer=trainer,
        datasource=target_dataset.DataSource,
        callbacks=[prompt_client_callbacks.ClientPayloadCallback],
        algorithm=fed_prompt_algorithm.Algorithm,
        personalized_model=stable_diffusion.Text2ImageSDPipeline,
        prompts=None,
    )

    server = fed_prompt_server.Server(
        model=prompt_learner,
        trainer=trainer,
        algorithm=fed_prompt_algorithm.Algorithm,
        personalized_model=None,
    )

    server.run(client)


if __name__ == "__main__":
    main()
