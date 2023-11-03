"""
An implementation of the SwAV algorithm.

Mathilde Caron, et al., Unsupervised Learning of Visual Features by Contrasting Cluster Assignments.
https://arxiv.org/abs/2006.09882, NeurIPS 2020.

Source code: https://github.com/facebookresearch/swav.
"""
from self_supervised_learning import ssl_client
from self_supervised_learning import ssl_trainer
from self_supervised_learning import ssl_datasources

import swav_model

from plato.servers import fedavg_personalized as personalized_server


def main():
    """The main running session for the SwAV algorithm."""
    client = ssl_client.Client(
        model=swav_model.SwaV,
        datasource=ssl_datasources.SSLDataSource,
        trainer=ssl_trainer.Trainer,
    )
    server = personalized_server.Server(
        model=swav_model.SwaV, trainer=ssl_trainer.Trainer
    )

    server.run(client)


if __name__ == "__main__":
    main()
