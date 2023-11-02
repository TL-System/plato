"""
The implementation for the SimCLR [1] method for personalized federated learning.

[1]. Ting Chen, et al., A Simple Framework for Contrastive Learning of Visual Representations, 
ICML 2020. https://arxiv.org/abs/2002.05709

The official code: https://github.com/google-research/simclr

The structure of our SimCLR and the classifier is the same as the ones used in
the work https://github.com/spijkervet/SimCLR.git.

"""
from pflbases import fedavg_personalized
from pflbases import ssl_datasources
from pflbases import ssl_client
from pflbases import ssl_trainer

from model import SimCLR


def main():
    """
    A personalized federated learning session for SimCLR approach.
    """
    trainer = ssl_trainer.Trainer
    client = ssl_client.Client(
        model=SimCLR,
        datasource=ssl_datasources.SSLDataSource,
        trainer=trainer,
    )
    server = fedavg_personalized.Server(model=SimCLR, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
