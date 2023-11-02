"""
The implementation for the FedEMA proposed by the work [1].

Zhuang, et.al, "Divergence-aware Federated Self-Supervised Learning", ICLR22.
https://arxiv.org/pdf/2204.04385.pdf.

"""

from pflbases import ssl_datasources
from pflbases import ssl_client

import fedema_server
import fedema_trainer
import fedema_callback
import fedema_model
import fedema_algorithm


def main():
    """Running the FedEMA approach."""
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
        algorithm=fedema_algorithm.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
