"""
The interface of applying the fedavg on different datasets as the
baseline.

"""

from plato.servers import fedavg
from plato.trainers import basic
from plato.algorithms import fedavg as fedavg_algo
from plato.clients import pers_simple


def main():
    """ An interface for running the fedavg method under the
        supervised learning setting.
    """

    algo = fedavg_algo.Algorithm
    trainer = basic.Trainer
    client = pers_simple.Client(algorithm=algo, trainer=trainer)
    server = fedavg.Server(algorithm=algo, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
