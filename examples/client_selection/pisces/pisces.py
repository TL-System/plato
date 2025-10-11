"""
Pisces: an asynchronous client selection and server aggregation algorithm.

Reference:

Z. Jiang, B. Wang, B. Li, B. Li. "Pisces: Efficient Federated Learning via Guided Asynchronous
Training," in Proceedings of ACM Symposium on Cloud Computing (SoCC), 2022.

URL: https://arxiv.org/abs/2206.09264
"""

import pisces_client
import pisces_server
import pisces_trainer


def main():
    """Pisces: an asynchronous client selection and server aggregation algorithm."""
    trainer = pisces_trainer.Trainer
    client = pisces_client.Client(trainer=trainer)
    server = pisces_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
