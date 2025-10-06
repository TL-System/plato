"""
A federated learning training session using the FedAdp algorithm.

Reference:

H. Wu, P. Wang. "Fast-Convergent Federated Learning with Adaptive Weighting," in IEEE Trans.
on Cognitive Communications and Networking (TCCN), 2021.

https://ieeexplore.ieee.org/abstract/document/9442814
"""

import fedadp_server


def main():
    """A federated learning training session using the FedAdp algorithm."""
    server = fedadp_server.Server()
    server.run()


if __name__ == "__main__":
    main()
