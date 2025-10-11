"""
An implementation of the SimSiam algorithm.

X. Chen, et al., "Exploring Simple Siamese Representation Learning."
https://arxiv.org/pdf/2011.10566.pdf

Source code: https://github.com/facebookresearch/simsiam or https://github.com/PatrickHua/SimSiam
"""

import simsiam_model
import simsiam_trainer

from plato.clients import self_supervised_learning as ssl_client
from plato.datasources import self_supervised_learning as ssl_datasource
from plato.servers import fedavg_personalized as personalized_server


def main():
    """
    A self-supervised federated learning session with SimSiam.
    """
    client = ssl_client.Client(
        model=simsiam_model.SimSiam,
        datasource=ssl_datasource.SSLDataSource,
        trainer=simsiam_trainer.Trainer,
    )
    server = personalized_server.Server(
        model=simsiam_model.SimSiam, trainer=simsiam_trainer.Trainer
    )

    server.run(client)


if __name__ == "__main__":
    main()
