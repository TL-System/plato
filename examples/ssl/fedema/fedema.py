"""
An implementation of the FedEMA algorithm.

Zhuang, et.al, "Divergence-aware Federated Self-Supervised Learning", ICLR22.
https://arxiv.org/pdf/2204.04385.pdf.

"""

import fedema_callback
import fedema_model
import fedema_server
import fedema_trainer

from plato.clients import self_supervised_learning as ssl_client
from plato.datasources import self_supervised_learning as ssl_datasource


def main():
    """
    A self-supervised federated learning session with FedEMA.
    """
    client = ssl_client.Client(
        model=fedema_model.BYOLModel,
        datasource=ssl_datasource.SSLDataSource,
        trainer=fedema_trainer.Trainer,
        callbacks=[
            fedema_callback.FedEMACallback,
        ],
    )
    server = fedema_server.Server(
        model=fedema_model.BYOLModel,
        trainer=fedema_trainer.Trainer,
    )

    server.run(client)


if __name__ == "__main__":
    main()
