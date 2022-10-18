"""
An asynchronous federated learning framework using Sirius.

Reference: 
Jiang, Z., Wang, W., Li, B., Li, B. (2022).
"Sirius: Efficient Federated Learning via Guided Asynchronous Training." 
Proceedings of ACM Symposium on Cloud Computing (SoCC).
"""
import pisces_client
import pisces_server
import pisces_trainer


def main():
    """A Plato federated learning training session using the Sirius algorithm."""
    trainer = pisces_trainer.Trainer
    client = pisces_client.Client(trainer=trainer)
    server = pisces_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
