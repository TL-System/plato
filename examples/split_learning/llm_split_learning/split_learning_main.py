"""
Use Split learning to finetune Huggingface large language model.
"""
import split_learning_trainer
from split_learning_llm_model import ServerModel, ClientModel
from split_learning_lora_algorithm import Algorithm as LoRAAlgorithm

from plato.servers.split_learning import Server
from plato.clients.split_learning import Client
from plato.config import Config


def main():
    """A Plato federated learning training session using the split learning algorithm."""
    if hasattr(Config().parameters, "lora"):
        client = Client(
            trainer=split_learning_trainer.Trainer,
            model=ClientModel,
            algorithm=LoRAAlgorithm,
        )
        server = Server(
            trainer=split_learning_trainer.Trainer,
            model=ServerModel,
            algorithm=LoRAAlgorithm,
        )
    else:
        client = Client(trainer=split_learning_trainer.Trainer, model=ClientModel)
        server = Server(trainer=split_learning_trainer.Trainer, model=ServerModel)
    server.run(client)


if __name__ == "__main__":
    main()
