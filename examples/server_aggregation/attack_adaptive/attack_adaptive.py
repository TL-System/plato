"""
A federated learning training session using the algorithm in the following unpublished
manuscript:

Ching Pui Wan, Qifeng Chen, "Robust Federated Learning with Attack-Adaptive Aggregation"

Unpublished
(https://arxiv.org/pdf/2102.05257.pdf)
"""

import attack_adaptive_server


def main():
    """A Plato federated learning training session using the attack-adaptive
    federation algorithm."""
    server = attack_adaptive_server.Server()
    server.run()


if __name__ == "__main__":
    main()
