"""
A federated unlearning algorithm to enables data holders to
proactively erase their data from a trained model.

Liu et al., "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid
Retraining," in Proc. INFOCOM, 2022.

Reference: https://arxiv.org/abs/2203.07320
"""

import fedunlearning_client
import fedunlearning_server
#import federaser_trainer


def main():
    """
    A naive retrain example used as fed unlearning baseline
    """
    #trainer = federaser_trainer.Trainer
    #client = fedunlearning_client.Client(trainer=trainer)
    #server = fedunlearning_server.Server(trainer=trainer)
    client = fedunlearning_client.Client()
    server = fedunlearning_server.Server()
    server.run(client)


if __name__ == "__main__":
    main()
