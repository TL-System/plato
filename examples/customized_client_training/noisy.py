"""
A federated learning training session using the Coreset method.
"""
import os
import coreset_trainer

from plato.servers import fedavg
from plato.clients import simple

def main():
    """ A Plato federated learning training session using the Coreset algorithm. """
    trainer = coreset_trainer.Trainer
    client = simple.Client(trainer=trainer)
    server = fedavg.Server(trainer=trainer)
    server.run(client)
if __name__ == "__main__":
    main()


