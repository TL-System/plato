"""
Knot: a clustered aggregation mechanism designed for federated unlearning.
"""

import knot_algorithm
import knot_client
import knot_server
import knot_trainer


def main():
    """
    Knot: a clustered aggregation mechanism designed for federated unlearning.
    """
    algorithm = knot_algorithm.Algorithm
    trainer = knot_trainer.Trainer
    client = knot_client.Client(algorithm=algorithm, trainer=trainer)
    server = knot_server.Server(algorithm=algorithm, trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
