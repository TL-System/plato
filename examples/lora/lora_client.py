"""
A federated learning client using LoRA fine-tuning.


To run this example:

python examples/lora/lora_client.py -c examples/lora/client.yml -i <client_id>
"""

import asyncio
import logging

from lora_utils import Algorithm, DataSource, LoraModel, Trainer

from plato.clients import simple


class Client(simple.Client):
    """A client using LoRA fine-tuning."""

    def __init__(self, model=None, datasource=None, trainer=None, algorithm=None):
        super().__init__(
            model=model, datasource=datasource, trainer=trainer, algorithm=algorithm
        )
        logging.info("A LoRA client has been initialized.")


def main():
    client = Client(
        model=LoraModel, datasource=DataSource, trainer=Trainer, algorithm=Algorithm
    )
    client.configure()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.start_client())


if __name__ == "__main__":
    main()
