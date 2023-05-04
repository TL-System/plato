"""
The implementation of Per-FedAvg method based on the plato's
pFL code.

Alireza Fallah, et.al, Personalized federated learning with theoretical guarantees:
A model-agnostic meta-learning approach, NeurIPS 2020.
https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html

Official code: None
Third-part code: https://github.com/jhoon-oh/FedBABU

"""

import perfedavg_trainer
import perfedavg_client

from examples.pfl.bases import fedavg_personalized


def main():
    """An interface for running the Per-FedAvg method under the
    supervised learning setting.
    """

    trainer = perfedavg_trainer.Trainer
    client = perfedavg_client.Client(trainer=trainer)
    server = fedavg_personalized.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
