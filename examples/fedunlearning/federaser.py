"""
FedEraser: A federated unlearning algorithm enabling client-level data removal from FL models.

Liu et al., "FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning
Models," in 2021 IEEE/ACM 29th International Symposium on Quality of Service (IWQoS 2021).

FedEraser has been proposed to be a significantly faster method of retraining the model with data
erasure requested.

Reference: https://ieeexplore.ieee.org/document/9521274
"""

import federaser_client
import federaser_server
import federaser_trainer

def main():
    """ The FedEraser algorithm. """
    trainer = federaser_trainer.Trainer
    client = federaser_client.Client(trainer=trainer)
    server = federaser_server.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
