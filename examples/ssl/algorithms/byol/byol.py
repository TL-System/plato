"""
An implementation of the BYOL algorithm.

Jean-Bastien Grill, et al., Bootstrap Your Own Latent A New Approach to Self-Supervised Learning.
https://arxiv.org/pdf/2006.07733.pdf.

Source code: https://github.com/lucidrains/byol-pytorch or https://github.com/sthalles/PyTorch-BYOL.
"""

from plato.servers import fedavg_personalized as personalized_server

from self_supervised_learning import ssl_client
from self_supervised_learning import ssl_datasources

from byol_model import BYOLModel
import byol_trainer


def main():
    """
    A personalized federated learning session for BYOL approach.
    """
    trainer = byol_trainer.Trainer
    client = ssl_client.Client(
        model=BYOLModel,
        datasource=ssl_datasources.SSLDataSource,
        trainer=trainer,
    )
    server = personalized_server.Server(model=BYOLModel, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
