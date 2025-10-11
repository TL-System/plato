"""
An implementation of the BYOL algorithm.

Jean-Bastien Grill, et al., Bootstrap Your Own Latent A New Approach to Self-Supervised Learning.
https://arxiv.org/pdf/2006.07733.pdf.

Source code: https://github.com/lucidrains/byol-pytorch or https://github.com/sthalles/PyTorch-BYOL.
"""

import byol_trainer
from byol_model import BYOLModel

from plato.clients import self_supervised_learning as ssl_client
from plato.datasources import self_supervised_learning as ssl_datasource
from plato.servers import fedavg_personalized as personalized_server


def main():
    """
    A self-supervised federated learning session with BYOL.
    """
    trainer = byol_trainer.Trainer
    client = ssl_client.Client(
        model=BYOLModel,
        datasource=ssl_datasource.SSLDataSource,
        trainer=trainer,
    )
    server = personalized_server.Server(model=BYOLModel, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
