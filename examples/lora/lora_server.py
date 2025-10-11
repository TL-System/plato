"""
A federated learning server using LoRA fine-tuning.

To run this example:

python examples/lora/lora_server.py -c examples/lora/server.yml
"""

import logging

from lora_utils import Algorithm, DataSource, LoraModel, Trainer

from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using LoRA fine-tuning."""

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
        logging.info("A LoRA server has been initialized.")

    def save_to_checkpoint(self):
        logging.info("Skipping checkpoint.")


def main():
    server = Server(
        model=LoraModel, datasource=DataSource, trainer=Trainer, algorithm=Algorithm
    )
    server.run()


if __name__ == "__main__":
    main()
