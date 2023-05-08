"""
A federated learning client using SCAFFOLD.

This version of SCAFFOLD will also support personalized federated learning because 
its client, trainer, and server are equipped with personalized components.

Additionally, the personalized variant of SCAFFOLD is that after training the global 
model with the federated paradigm, each client will finetune the received global model 
based on local samples to generate the personalized model.

See the `scaffold_finetune_MNIST_lenet5_noniid.yml` for how to set hyper-parameters
for personalization.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
from scaffold_callback import ScaffoldCallback

import scaffold_client
import scaffold_server
import scaffold_trainer


def main():
    """A Plato federated learning training session using the SCAFFOLD algorithm."""
    trainer = scaffold_trainer.Trainer
    client = scaffold_client.Client(trainer=trainer, callbacks=[ScaffoldCallback])
    server = scaffold_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
