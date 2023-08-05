import feddyn_trainer

from plato.servers import fedavg
from plato.clients import simple


def main():
    """A Plato federated learning training session using FedProx."""
    trainer = feddyn_trainer.Trainer
    client = simple.Client(trainer=trainer)
    server = fedavg.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
