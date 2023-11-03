"""
Implementation of Textual Inversion [1] under the federated learning,
referred to as Federated Textual Inversion (FedTI).

[1]. Rinon, et al., An Image is Worth One Word: Personalizing Text-to-Image Generation using 
    Textual Inversion, ICLR23.

Textual inversion is a technique that allows us to add new styles or objects 
to text-to-image models without modifying the underlying model.

"""


from plato.algorithms import fedavg as fedavg_algorithm


from plato.config import Config

import generation_model
import fedti_server
import fedti_client
import fedti_trainer
import flowers102


datasets = {"Flowers102OneClass": flowers102}


def main():
    """
    A personalized federated learning session for generation approach.
    """
    data_name = Config().data.datasource
    target_dataset = datasets[data_name]

    trainer = fedti_trainer.Trainer

    # define the prompt learner for t2i generation
    prompt_learner = generation_model.GenerationPromptLearner

    client = fedti_client.Client(
        model=prompt_learner,
        trainer=trainer,
        datasource=target_dataset.DataSource,
        algorithm=fedavg_algorithm.Algorithm,
    )

    server = fedti_server.Server(
        model=prompt_learner, trainer=trainer, algorithm=fedavg_algorithm.Algorithm
    )

    server.run(client)


if __name__ == "__main__":
    main()
