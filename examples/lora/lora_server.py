"""
This example uses a very simple model to show how the model and the server
be customized in Plato and executed in a standalone fashion.

To run this example:

python examples/customized/custom_server.py -c examples/customized/server.yml
"""
import logging

from plato.servers import fedavg

from lora_utils import LoraModel, DataSource, Trainer, Algorithm


class CustomServer(fedavg.Server):
    """A custom federated learning server."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        logging.info("A custom server has been initialized.")

    def save_to_checkpoint(self):
        logging.info("Skipping checkpoint.")


def main():
    server = CustomServer(
        model=LoraModel, datasource=DataSource, trainer=Trainer, algorithm=Algorithm
    )
    server.run()


if __name__ == "__main__":
    main()
