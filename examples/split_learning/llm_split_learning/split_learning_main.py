"""
Finetune HuggingFace large language models using split learning.
"""

import split_learning_trainer
from split_learning_llm_model import ClientModel, ServerModel
from split_learning_lora_algorithm import Algorithm as LoRAAlgorithm

from plato.clients.split_learning import Client
from plato.config import Config
from plato.servers.split_learning import Server


def main():
    """A Plato federated learning training session using the split learning algorithm."""

    algorithm = LoRAAlgorithm if hasattr(Config().parameters, "lora") else None

    client = Client(
        trainer=split_learning_trainer.Trainer, model=ClientModel, algorithm=algorithm
    )
    server = Server(
        trainer=split_learning_trainer.Trainer, model=ServerModel, algorithm=algorithm
    )
    server.run(client)


if __name__ == "__main__":
    main()
