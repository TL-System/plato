"""
The implementation of FedPer method based on the plato's pFL code.

Manoj Ghuhan Arivazhagan, et al., Federated learning with personalization layers, 2019.
https://arxiv.org/abs/1912.00818

Official code: None
Third-part code: https://github.com/jhoon-oh/FedBABU
"""


from pflbases import fedavg_personalized
from pflbases import fedavg_partial
import fedper_trainer
from plato.clients import simple


def main():
    """
    A personalized federated learning session for FedPer approach.
    """
    trainer = fedper_trainer.Trainer
    client = simple.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )
    server = fedavg_personalized.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
