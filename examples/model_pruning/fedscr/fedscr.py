"""
A federated learning training session using FedSCR.

X. Wu, X. Yao and C. -L. Wang,
"FedSCR: Structure-Based Communication Reduction for Federated Learning,"
in IEEE Transactions on Parallel and Distributed Systems,
vol. 32, no. 7, pp. 1565-1577, 1 July 2021, doi: 10.1109/TPDS.2020.3046250.
"""

import fedscr_client
import fedscr_server
import fedscr_trainer


def main():
    """A Plato federated learning training session using the FedSCR algorithm."""
    trainer = fedscr_trainer.Trainer
    client = fedscr_client.Client(trainer=trainer)
    server = fedscr_server.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
