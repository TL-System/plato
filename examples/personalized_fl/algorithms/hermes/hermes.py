"""
A federated learning training session using Hermes

A. Li, J. Sun, P. Li, Y. Pu, H. Li, and Y. Chen, 
“Hermes: An Efficient Federated Learning Framework for Heterogeneous Mobile Clients,”
in Proc. 27th Annual International Conference on Mobile Computing and Networking (MobiCom), 2021.
"""

import hermes_trainer
import hermes_server
from plato.clients import simple


def main():
    """A Plato federated learning training session using the Hermes algorithm."""
    trainer = hermes_trainer.Trainer
    server = hermes_server.Server(trainer=trainer)
    client = simple.Client(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
