"""
The implementation for the SwAV [1] method.

[1]. Mathilde Caron, et al., Unsupervised Learning of Visual Features by Contrasting Cluster Assignments.
https://arxiv.org/abs/2006.09882, NeurIPS 2020.

Source code: https://github.com/facebookresearch/swav
"""

from pflbases import fedavg_personalized

from pflbases.models import SSL
from pflbases import ssl_client
from pflbases import ssl_trainer
from pflbases import ssl_datasources


import swav_model


def main():
    """
    A pFL session for SwaV approach.
    """
    trainer = ssl_trainer.Trainer
    client = ssl_client.Client(
        model=swav_model.SwaV,
        datasource=ssl_datasources.SSLDataSource,
        trainer=trainer,
    )
    server = fedavg_personalized.Server(model=SSL.SwaV, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
