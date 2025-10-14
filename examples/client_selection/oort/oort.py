"""
A federated learning training session using Oort with strategy pattern.

This is the updated version using the strategy-based API instead of inheritance.

Reference:

F. Lai, X. Zhu, H. V. Madhyastha and M. Chowdhury, "Oort: Efficient Federated Learning via
Guided Participant Selection," in USENIX Symposium on Operating Systems Design and Implementation
(OSDI 2021), July 2021.
"""

import oort_client
import oort_server
import oort_trainer


def main():
    """A Plato federated learning training session using Oort strategy."""
    trainer = oort_trainer.Trainer
    client = oort_client.Client(trainer=trainer)
    server = oort_server.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
