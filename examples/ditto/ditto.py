"""
The implementation of Ditto method based on the pFL framework of Plato.

Tian Li, et.al, Ditto: Fair and robust federated learning through personalization, 2021:
 https://proceedings.mlr.press/v139/li21h.html

Official code: https://github.com/s-huu/Ditto
Third-part code: https://github.com/lgcollins/FedRep

"""

# import ditto_trainer
import ditto_trainer_v2 as ditto_trainer
import ditto_client

from plato.servers import fedavg_personalized


def main():
    """An interface for running the APFL method."""

    trainer = ditto_trainer.Trainer
    client = ditto_client.Client(trainer=trainer)
    server = fedavg_personalized.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
