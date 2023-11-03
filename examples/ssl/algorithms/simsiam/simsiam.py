"""
An implementation of the SimSiam algorithm.

Xinlei Chen, et al., Exploring Simple Siamese Representation Learning.
https://arxiv.org/pdf/2011.10566.pdf

Source code: https://github.com/facebookresearch/simsiam or https://github.com/PatrickHua/SimSiam
"""

from self_supervised_learning import ssl_client
from self_supervised_learning import ssl_datasources

import simsiam_trainer
import simsiam_model

from plato.servers import fedavg_personalized as personalized_server


def main():
    """The main running session for the SimSiam approach."""
    client = ssl_client.Client(
        model=simsiam_model.SimSiam,
        datasource=ssl_datasources.SSLDataSource,
        trainer=simsiam_trainer.Trainer,
    )
    server = personalized_server.Server(
        model=simsiam_model.SimSiam, trainer=simsiam_trainer.Trainer
    )

    server.run(client)


if __name__ == "__main__":
    main()
