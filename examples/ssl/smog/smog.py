"""
An implementation of the SMoG algorithm.

B. Pang, et al., "Unsupervised Visual Representation Learning by Synchronous Momentum Grouping," ECCV, 2022.

https://arxiv.org/pdf/2006.07733.pdf
"""

import smog_model
import smog_trainer

from plato.clients import self_supervised_learning as ssl_client
from plato.datasources import self_supervised_learning as ssl_datasource
from plato.servers import fedavg_personalized as personalized_server


def main():
    """
    A self-supervised federated learning session with SMoG.
    """
    client = ssl_client.Client(
        model=smog_model.SMoG,
        datasource=ssl_datasource.SSLDataSource,
        trainer=smog_trainer.Trainer,
    )
    server = personalized_server.Server(
        model=smog_model.SMoG,
        trainer=smog_trainer.Trainer,
    )

    server.run(client)


if __name__ == "__main__":
    main()
