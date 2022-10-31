"""
Reference:
J. Xu, et al. "FedCM: Federated Learning with Client-level Momentum," found in papers/.
"""


# configure dataset and model via the configutation files
import fedcm_client
import fedcm_trainer
import fedcm_server
import fedcm_algorithm

def main():
    """ A Plato federated learning training session using Adaptive Synchronization Frequency. """
    trainer = fedcm_trainer.Trainer
    algorithm = fedcm_algorithm.Algorithm
    client = fedcm_client.Client(algorithm=algorithm, trainer=trainer)
    server = fedcm_server.Server(algorithm=algorithm, trainer=trainer)
    server.run(client)

if __name__ == "__main__":
    main()
