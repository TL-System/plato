"""
Implementation of our Calibre algorithm.
"""
from ssl import ssl_datasources
from ssl import ssl_client


import calibre_model
import calibre_trainer
import calibre_callback
import calibre_server


def main():
    """
    A personalized federated learning session for SimCLR approach.
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
