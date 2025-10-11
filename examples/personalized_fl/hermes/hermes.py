"""
A federated learning training session using Hermes.

A. Li, J. Sun, P. Li, Y. Pu, H. Li, and Y. Chen,
“Hermes: An Efficient Federated Learning Framework for Heterogeneous Mobile
Clients,” in Proc. 27th Annual International Conference on Mobile Computing and
Networking (MobiCom), 2021.
"""

import hermes_server
import hermes_trainer
from hermes_callback import HermesCallback

from plato.clients import fedavg_personalized as personalized_client


def main():
    """A Plato federated learning training session using the Hermes algorithm."""
    trainer = hermes_trainer.Trainer

    client = personalized_client.Client(trainer=trainer, callbacks=[HermesCallback])
    server = hermes_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
