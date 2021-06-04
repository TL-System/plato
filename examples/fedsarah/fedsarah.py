import os

os.environ['config_file'] = 'fedsarah_MNIST_lenet5.yml'

import fedsarah_client
import fedsarah_server
import fedsarah_trainer

def main():
    """ A Plato federated learning training session using the FedNova algorithm. """
    trainer = fedsarah_trainer.Trainer()

    client = fedsarah_client.Client(trainer=trainer)
    server = fedsarah_server.Server(trainer=trainer)
    server.run(client)

if __name__ == "__main__":
    main()
