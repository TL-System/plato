"""
A federated learning training session using FedBuff.

Reference:

Nguyen, J., Malik, K., Zhan, H., et al., "Federated Learning with Buffered Asynchronous Aggregation,
" in Proc. International Conference on Artificial Intelligence and Statistics (AISTATS 2022).

https://proceedings.mlr.press/v151/nguyen22b/nguyen22b.pdf
"""

import fedbuff_server


def main():
    """A Plato federated learning training session using FedBuff."""
    server = fedbuff_server.Server()
    server.run()


if __name__ == "__main__":
    main()
