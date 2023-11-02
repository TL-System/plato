"""
The implementation for the SwAV [1] method.

Reference:
[1]. Mathilde Caron, et al., Unsupervised Learning of Visual Features by Contrasting Cluster Assignments.
https://arxiv.org/abs/2006.09882, NeurIPS 2020.

Source code: https://github.com/facebookresearch/swav
"""

from plato.servers import fedavg_personalized as personalized_server

from ssl import ssl_client
from ssl import ssl_trainer
from ssl import ssl_datasources


import swav_model


def main():
    """
    A pFL session for SwaV approach.
    """
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
