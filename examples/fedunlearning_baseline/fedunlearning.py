"""
A federated unlearning algorithm to enables data holders to proactively erase their data from a trained model.

Liu et al., "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid
Retraining," in Proc. INFOCOM, 2022.

Reference: https://arxiv.org/abs/2203.07320
"""

import fedunlearning_client
import fedunlearning_server


def main():
    """
    Federated unlearning with retraining from scratch.
    """
    client = fedunlearning_client.Client()
    server = fedunlearning_server.Server()
    server.run(client)


if __name__ == "__main__":
    main()
