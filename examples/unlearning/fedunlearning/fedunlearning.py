"""
Federated unlearning allows clients to proactively erase their data from a trained model. The model
will be retrained from scratch during the unlearning process.

This generic retraining workflow is related to the following paper. The historical
AdaHessian variant is preserved in archives/retired/adahessian/README.md:

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
    client = fedunlearning_client.create_client()
    server = fedunlearning_server.Server()
    server.run(client)


if __name__ == "__main__":
    main()
