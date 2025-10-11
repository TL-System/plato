"""
Personalized federated learning using MAML algorithm

Reference:
Finn, Chelsea, Pieter Abbeel, and Sergey Levine.
"Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks."
International Conference on Machine Learning. PMLR, 2017.
http://proceedings.mlr.press/v70/finn17a/finn17a.pdf
"""

import fl_maml_client
import fl_maml_server
import fl_maml_trainer


def main():
    """A Plato federated learning training session using the MAML algorithm."""
    trainer = fl_maml_trainer.Trainer
    client = fl_maml_client.Client(trainer=trainer)
    server = fl_maml_server.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
