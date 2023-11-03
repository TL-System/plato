"""
An implementation of the SMoG algorithm.

Bo Pang, et al., Unsupervised Visual Representation Learning by Synchronous Momentum Grouping.
ECCV, 2022. https://arxiv.org/pdf/2006.07733.pdf.
"""

from self_supervised_learning import ssl_client
from self_supervised_learning import ssl_datasources

import smog_trainer
import smog_model

from plato.servers import fedavg_personalized as personalized_server


def main():
    """
    A personalized federated learning session with SMoG.
    """
    client = ssl_client.Client(
        model=smog_model.SMoG,
        datasource=ssl_datasources.SSLDataSource,
        trainer=smog_trainer.Trainer,
    )
    server = personalized_server.Server(
        model=smog_model.SMoG,
        trainer=smog_trainer.Trainer,
    )

    server.run(client)


if __name__ == "__main__":
    main()
