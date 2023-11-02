"""
The implementation of Per-FedAvg method based on the plato's
pFL code.

Reference
Alireza Fallah, et al., Personalized federated learning with theoretical guarantees:
A model-agnostic meta-learning approach, NeurIPS 2020.
https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html

Third-party code: https://github.com/jhoon-oh/FedBABU

"""

import perfedavg_trainer

from plato.servers import fedavg_personalized as personalized_server
from plato.clients import fedavg_personalized as personalized_client

def main():
    """
    A personalized federated learning session for PerFedAvg approach.
    """
    trainer = perfedavg_trainer.Trainer
    client = personalized_client.Client(trainer=trainer)
    server = personalized_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
