"""
Implementation of our Calibre algorithm.
"""

from pflbases import fedavg_partial

from pflbases.client_callbacks import local_completion_callbacks

from pflbases import ssl_datasources


import calibre_model
import calibre_trainer
import calibre_client
import calibre_callback
import calibre_server


def main():
    """
    A personalized federated learning sesstion for SimCLR approach.
    """
    trainer = calibre_trainer.Trainer
    client = calibre_client.Client(
        model=calibre_model.CalibreNet,
        datasource=ssl_datasources.TransformedDataSource,
        personalized_datasource=ssl_datasources.TransformedDataSource,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            local_completion_callbacks.ClientModelLocalCompletionCallback,
            calibre_callback.CalibreCallback,
        ],
    )
    server = calibre_server.Server(
        model=calibre_model.CalibreNet,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
