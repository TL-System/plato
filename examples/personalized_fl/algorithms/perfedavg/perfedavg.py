"""
The implementation of Per-FedAvg method based on the plato's
pFL code.

Reference
Alireza Fallah, et al., Personalized federated learning with theoretical guarantees:
A model-agnostic meta-learning approach, NeurIPS 2020.
https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html

Third-part code: https://github.com/jhoon-oh/FedBABU

"""

from pflbases import fedavg_personalized
from pflbases import fedavg_partial
from pflbases import fedavg_personalized_client

import perfedavg_trainer


def main():
    """
    A personalized federated learning session for PerFedAvg approach.
    """
    trainer = perfedavg_trainer.Trainer
    client = fedavg_personalized_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )
    server = fedavg_personalized.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
