"""
An implementation of the FedEMA algorithm.

Zhuang, et.al, "Divergence-aware Federated Self-Supervised Learning", ICLR22.
https://arxiv.org/pdf/2204.04385.pdf.

"""
from self_supervised_learning import ssl_datasources
from self_supervised_learning import ssl_client

import fedema_server
import fedema_trainer
import fedema_callback
import fedema_model


def main():
    """
    A personalized federated learning session with FedEMA.
    """
    client = ssl_client.Client(
        model=fedema_model.BYOLModel,
        datasource=ssl_datasources.SSLDataSource,
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
