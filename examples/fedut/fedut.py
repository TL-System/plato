"""
A federated learning training session using utility evaluation.
"""
import os

os.environ['config_file'] = './fedut_MNIST_lenet5.yml'

#import fedut_client
#import fedut_server
from plato.servers import fedavg
from plato.clients import simple
import fedut_trainer


def main():
    """ A Plato federated learning training session using the FedNova algorithm. """
    # client = fedut_client.Client()
    # server = fedut_server.Server()
    trainer = fedut_trainer.Trainer()
    client = simple.Client(trainer=trainer)
    server = fedavg.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
