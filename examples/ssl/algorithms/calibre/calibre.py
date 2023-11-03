"""
An implementation of the Calibre algorithm.
"""
from plato.clients import self_supervised_learning as ssl_client
from plato.datasources import self_supervised_learning as ssl_datasource

import calibre_model
import calibre_trainer
import calibre_callback
import calibre_server


def main():
    """
    A self-supervised federated learning session with Calibre.
    """
    client = ssl_client.Client(
        model=calibre_model.CalibreNet,
        datasource=ssl_datasource.SSLDataSource,
        trainer=calibre_trainer.Trainer,
        callbacks=[
            calibre_callback.CalibreCallback,
        ],
    )
    server = calibre_server.Server(
        model=calibre_model.CalibreNet,
        trainer=calibre_trainer.Trainer,
    )

    server.run(client)


if __name__ == "__main__":
    main()
