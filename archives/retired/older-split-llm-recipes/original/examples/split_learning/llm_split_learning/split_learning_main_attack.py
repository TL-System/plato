"""
Finetune HuggingFace large language models using split learning and implement the Unsplit attack.

Reference of Unsplit  attack:
E. Erdoğan, A. Küpçü, and A. Çiçek, "UnSplit: Data-Oblivious Model Inversion,
Model Stealing, and Label Inference Attacks against Split Learning," in
proceedings of the 21st Workshop on Privacy in the Electronic Society (WPES'22),
Association for Computing Machinery, New York, NY, USA,
115–124. https://doi.org/10.1145/3559613.3563201.
"""

from split_learning_llm_model import ClientModel
from split_learning_llm_model_attack import ServerModelCurious
from split_learning_lora_algorithm import Algorithm as LoRAAlgorithm
from split_learning_server_attack import CuriousServer
from split_learning_trainer_attack import CuriousTrainer, HonestTrainer

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
