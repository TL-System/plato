"""
The implementation for the SimCLR [1] method for personalized federated learning.

[1]. Ting Chen, et.al., A Simple Framework for Contrastive Learning of Visual Representations, 
ICML 2020. https://arxiv.org/abs/2002.05709

The official code: https://github.com/google-research/simclr

The structure of our SimCLR and the classifier is the same as the ones used in
the work https://github.com/spijkervet/SimCLR.git.

"""

import simclr_model
import simclr_client
import simclr_trainer

from plato.servers import fedavg_personalized


def main():
    """A Plato personalized federated learning training session using the SimCLR approach."""

    trainer = simclr_trainer.Trainer
    simclr_approach = simclr_model.SimCLR
    client = simclr_client.Client(model=simclr_approach, trainer=trainer)
    server = fedavg_personalized.Server(model=simclr_approach, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
