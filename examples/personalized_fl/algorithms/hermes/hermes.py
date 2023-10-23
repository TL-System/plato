"""
A federated learning training session using Hermes

A. Li, J. Sun, P. Li, Y. Pu, H. Li, and Y. Chen, 
“Hermes: An Efficient Federated Learning Framework for Heterogeneous Mobile Clients,”
in Proc. 27th Annual International Conference on Mobile Computing and Networking (MobiCom), 2021.
"""

from pflbases import fedavg_partial
from pflbases import personalized_client

from pflbases.client_callbacks import base_callbacks

from hermes_callback import HermesCallback
import hermes_trainer
import hermes_server


def main():
    """A Plato federated learning training session using the Hermes algorithm."""
    trainer = hermes_trainer.Trainer

    client = personalized_client.Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[base_callbacks.ClientPayloadCallback, HermesCallback],
    )
    server = hermes_server.Server(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
