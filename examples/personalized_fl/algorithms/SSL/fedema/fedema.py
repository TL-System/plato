"""
The implementation for the FedEMA proposed by the work [1].

Zhuang, et.al, "Divergence-aware Federated Self-Supervised Learning", ICLR22.
https://arxiv.org/pdf/2204.04385.pdf.

No official code.
"""

from pflbases import ssl_datasources
from pflbases.models import SSL

from pflbases import fedavg_partial
from pflbases import ssl_client

import fedema_server
import fedema_trainer
import fedema_callback


def main():
    """Running the FedEMA approach."""
    trainer = fedema_trainer.Trainer
    client = ssl_client.Client(
        model=SSL.BYOL,
        datasource=ssl_datasources.TransformedDataSource,
        personalized_datasource=ssl_datasources.TransformedDataSource,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            fedema_callback.FedEMACallback,
        ],
    )
    server = fedema_server.Server(
        model=SSL.BYOL,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
