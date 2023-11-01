"""
The implementation for the SimCLR [1] method for personalized federated learning.

[1]. Ting Chen, et al., A Simple Framework for Contrastive Learning of Visual Representations, 
ICML 2020. https://arxiv.org/abs/2002.05709

The official code: https://github.com/google-research/simclr

The structure of our SimCLR and the classifier is the same as the ones used in
the work https://github.com/spijkervet/SimCLR.git.

"""


from pflbases import fedavg_personalized_server
from pflbases import fedavg_partial

from pflbases.models import SSL
from pflbases import ssl_datasources
from pflbases import ssl_client
from pflbases import ssl_trainer


def main():
    """
    A personalized federated learning session for SimCLR approach.
    """
    trainer = ssl_trainer.Trainer
    client = ssl_client.Client(
        model=SSL.SimCLR,
        datasource=ssl_datasources.TransformedDataSource,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )
    server = fedavg_personalized_server.Server(model=SSL.SimCLR, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
