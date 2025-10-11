"""
An implementation of the SimCLR algorithm.

T. Chen, et al., "A Simple Framework for Contrastive Learning of Visual Representations," ICML 2020.

https://arxiv.org/abs/2002.05709

Source code: https://github.com/google-research/simclr or https://github.com/spijkervet/SimCLR.git.

"""

from simclr_model import SimCLRModel

from plato.clients import self_supervised_learning as ssl_client
from plato.datasources import self_supervised_learning as ssl_datasource
from plato.servers import fedavg_personalized as personalized_server


def main():
    """
    A self-supervised federated learning session with SimCLR.
    """
    client = ssl_client.Client(
        model=SimCLRModel, datasource=ssl_datasource.SSLDataSource
    )
    server = personalized_server.Server(model=SimCLRModel)

    server.run(client)


if __name__ == "__main__":
    main()
