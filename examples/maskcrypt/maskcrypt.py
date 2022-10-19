"""
Customize the inbound and outbound processors through client callbacks.
"""
from maskcrypt_callbacks import MaskCryptCallback

import maskcrypt_trainer
import maskcrypt_client
import maskcrypt_server


def main():
    """A Plato federated learning training session using CustomizeProcessorCallback."""
    trainer = maskcrypt_trainer.Trainer
    client = maskcrypt_client.Client(trainer=trainer, callbacks=[MaskCryptCallback])
    server = maskcrypt_server.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
