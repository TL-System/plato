"""
The implementation of Per-FedAvg method based on the plato's
pFL code.

Reference
Alireza Fallah, et al., Personalized federated learning with theoretical guarantees:
A model-agnostic meta-learning approach, NeurIPS 2020.
https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html

Third-party code: https://github.com/jhoon-oh/FedBABU

"""

from pflbases import fedavg_personalized
from pflbases import fedavg_personalized
from pflbases import fedavg_personalized

import perfedavg_trainer


def main():
    """
    A personalized federated learning session for PerFedAvg approach.
    """
    trainer = perfedavg_trainer.Trainer
    client = fedavg_personalized.Client(
        trainer=trainer,
        algorithm=fedavg_personalized.Algorithm,
    )
    server = fedavg_personalized.Server(
        trainer=trainer,
        algorithm=fedavg_personalized.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
