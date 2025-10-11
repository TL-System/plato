"""
A federated learning training session with LoRA fine-tuning.
"""

import lora_client
import lora_server
from lora_utils import Algorithm, DataSource, LoraModel, Trainer


def main():
    """A Plato federated learning training session with LoRA fine-tuning."""
    client = lora_client.Client(
        model=LoraModel, datasource=DataSource, trainer=Trainer, algorithm=Algorithm
    )
    server = lora_server.Server(
        model=LoraModel, datasource=DataSource, trainer=Trainer, algorithm=Algorithm
    )
    server.run(client)


if __name__ == "__main__":
    main()
