"""
A federated learning training session using Oort.

Reference:

F. Lai, X. Zhu, H. V. Madhyastha and M. Chowdhury, "Oort: Efficient Federated Learning via
Guided Participant Selection," in USENIX Symposium on Operating Systems Design and Implementation
(OSDI 2021), July 2021.
"""

import oort_server
import oort_trainer
import oort_client


def main():
    """A Plato federated learning training session using Oort for client selection."""
    trainer = oort_trainer.Trainer
    client = oort_client.Client(trainer=trainer)
    server = oort_server.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
