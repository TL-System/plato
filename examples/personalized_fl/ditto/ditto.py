"""
The implementation of Ditto method based on the pFL framework of Plato.

Reference:
Tian Li, et al., Ditto: Fair and robust federated learning through personalization, 2021:
 https://proceedings.mlr.press/v139/li21h.html

Official code: https://github.com/litian96/ditto
Third-party code: https://github.com/lgcollins/FedRep

"""

import ditto_trainer

from plato.servers import fedavg_personalized as personalized_server
from plato.clients import fedavg_personalized as personalized_client

def main():
    """
    A personalized federated learning session for Ditto approach.
    """
    trainer = ditto_trainer.Trainer
    client = personalized_client.Client(trainer=trainer)
    server = personalized_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
