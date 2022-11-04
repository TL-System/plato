"""
A communication-efficient federated learning training session
using qfed, which is a modified version of federated qsgd.

Note that if we want to use this, we need to modify the plato
as a new local pip package!!!
"""

import qfed_trainer
import qfed_algorithm
import qfed_client
import qfed_server


def main():
    """A Plato federated learning training session using the qfed algorithm."""
    trainer = qfed_trainer.Trainer
    algorithm = qfed_algorithm.Algorithm
    client = qfed_client.Client(algorithm=algorithm, trainer=trainer)
    server = qfed_server.Server(algorithm=algorithm, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
