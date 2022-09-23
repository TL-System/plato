"""
An asynchronous federated learning framework using Sirius.

Reference: 
Jiang, Z., Wang, W., Li, B., Li, B. (2022).
"Sirius: Efficient Federated Learning via Guided Asynchronous Training." 
Proceedings of ACM Symposium on Cloud Computing (SoCC).
"""
import sirius_client
import sirius_server
import sirius_trainer


def main():
    """A Plato federated learning training session using the SCAFFOLD algorithm."""
    trainer = sirius_trainer.Trainer
    client = sirius_client.Client(trainer=trainer)
    server = sirius_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
