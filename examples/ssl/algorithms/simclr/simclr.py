"""
The implementation for the SimCLR [1] method for personalized federated learning.

[1]. Ting Chen, et al., A Simple Framework for Contrastive Learning of Visual Representations, 
ICML 2020. https://arxiv.org/abs/2002.05709

The official code: https://github.com/google-research/simclr

The structure of our SimCLR and the classifier is the same as the ones used in
the work https://github.com/spijkervet/SimCLR.git.

"""
from self_supervised_learning import ssl_datasources
from self_supervised_learning import ssl_client
from self_supervised_learning import ssl_trainer

from simclr_model import SimCLRModel

from plato.servers import fedavg_personalized as personalized_server


def main():
    """
    The main running session for the SimCLR approach.
    """
    client = ssl_client.Client(
        model=SimCLRModel,
        datasource=ssl_datasources.SSLDataSource,
        trainer=ssl_trainer.Trainer,
    )
    server = personalized_server.Server(model=SimCLRModel, trainer=ssl_trainer.Trainer)

    server.run(client)


if __name__ == "__main__":
    main()
