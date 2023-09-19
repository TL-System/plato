"""
A federated learning training session with the honest-but-curious server.
The server can analyze periodic gradients from certain clients to
perform the gradient leakage attacks and reconstruct the training data of the victim clients.
"""
import dlg_client
import dlg_server
import dlg_trainer


def main():
    """A Plato federated learning training session with the honest-but-curious server."""
    trainer = dlg_trainer.Trainer
    client = dlg_client.Client(trainer=trainer)
    server = dlg_server.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
