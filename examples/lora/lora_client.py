""" 
An example for running Plato with custom clients. 

To run this example:

python examples/customized/custom_client.py -c examples/customized/client.yml -i <client_id>
"""

import asyncio
import logging

from plato.clients import simple

from lora_utils import LoraModel, DataSource, Trainer, Algorithm


class CustomClient(simple.Client):
    """An example for customizing the client."""

    def __init__(self, model=None, datasource=None, trainer=None, algorithm=None):
        super().__init__(
            model=model, datasource=datasource, trainer=trainer, algorithm=algorithm
        )
        logging.info("A customized client has been initialized.")


def main():
    client = CustomClient(
        model=LoraModel, datasource=DataSource, trainer=Trainer, algorithm=Algorithm
    )
    client.configure()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.start_client())


if __name__ == "__main__":
    main()
