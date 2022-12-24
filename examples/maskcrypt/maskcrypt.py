"""
MaskCrypt: Federated Learning with Selective Homomorphic Encryption.
"""
import maskcrypt_trainer
import maskcrypt_client
import maskcrypt_server

from maskcrypt_callbacks import MaskCryptCallback


def main():
    """A Plato federated learning training session using selective homomorphic encryption."""
    trainer_callbacks = maskcrypt_trainer.get()
    client = maskcrypt_client.Client(
        callbacks=[MaskCryptCallback], trainer_callbacks=trainer_callbacks
    )
    server = maskcrypt_server.Server()
    server.run(client)


if __name__ == "__main__":
    main()
