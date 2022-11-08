"""
A federated learning training session using split learning.

Reference:

Vepakomma, et al., "Split Learning for Health: Distributed Deep Learning without Sharing
Raw Patient Data," in Proc. AI for Social Good Workshop, affiliated with ICLR 2018.

https://arxiv.org/pdf/1812.00564.pdf

Chopra, Ayush, et al. "AdaSplit: Adaptive Trade-offs for Resource-constrained Distributed 
Deep Learning." arXiv preprint arXiv:2112.01637 (2021).

https://arxiv.org/pdf/2112.01637.pdf
"""

import split_learning_algorithm
import split_learning_client
import split_learning_server
import split_learning_trainer


def main():
    """A Plato federated learning training session using the split learning algorithm."""
    trainer = split_learning_trainer.Trainer
    algorithm = split_learning_algorithm.Algorithm
    client = split_learning_client.Client(algorithm=algorithm, trainer=trainer)
    server = split_learning_server.Server(algorithm=algorithm, trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
