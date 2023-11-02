"""
Implementation of our Calibre algorithm.
"""


from pflbases import ssl_datasources
from pflbases import ssl_client


import calibre_model
import calibre_trainer
import calibre_callback
import calibre_server


def main():
    """
    A personalized federated learning session for SimCLR approach.
    """
    trainer = calibre_trainer.Trainer
    client = ssl_client.Client(
        model=calibre_model.CalibreNet,
        datasource=ssl_datasources.SSLDataSource,
        trainer=trainer,
        callbacks=[
            calibre_callback.CalibreCallback,
        ],
    )
    server = calibre_server.Server(
        model=calibre_model.CalibreNet,
        trainer=trainer,
    )

    server.run(client)


if __name__ == "__main__":
    main()
