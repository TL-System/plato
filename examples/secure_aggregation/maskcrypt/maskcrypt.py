"""
MaskCrypt: Federated Learning with Selective Homomorphic Encryption.
"""

import maskcrypt_client
import maskcrypt_server
import maskcrypt_trainer
from maskcrypt_callbacks import MaskCryptCallback


def main():
    """A Plato federated learning training session using selective homomorphic encryption."""
    trainer = maskcrypt_trainer.Trainer
    client = maskcrypt_client.Client(trainer=trainer, callbacks=[MaskCryptCallback])
    server = maskcrypt_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
