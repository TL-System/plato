"""
An implementation of the Calibre algorithm.
"""
from self_supervised_learning import ssl_datasources
from self_supervised_learning import ssl_client

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
        datasource=ssl_datasources.SSLDataSource,
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
