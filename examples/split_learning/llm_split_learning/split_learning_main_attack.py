"""
Finetune HuggingFace large language models using split learning.
"""
from split_learning_trainer_attack import CuriousTrainer, HonestTrainer
from split_learning_llm_model import ClientModel
from split_learning_llm_model_attack import ServerModelCurious
from split_learning_lora_algorithm import Algorithm as LoRAAlgorithm
from split_learning_server_attack import CuriousServer

from plato.clients.split_learning import Client
from plato.config import Config


def main():
    """A Plato federated learning training session using the split learning algorithm."""

    algorithm = LoRAAlgorithm if hasattr(Config().parameters, "lora") else None

    client = Client(trainer=HonestTrainer, model=ClientModel, algorithm=algorithm)
    server = CuriousServer(
        trainer=CuriousTrainer,
        model=ServerModelCurious,
        algorithm=algorithm,
    )
    server.run(client)


if __name__ == "__main__":
    main()
