"""
The implementation of the work:

Reference:
Fallah, Alireza, Aryan Mokhtari, and Asuman Ozdaglar.
"Personalized Federated Learning with Theoretical Guarantees: \
    A Model-Agnostic Meta-Learning Approach."
Advances in Neural Information Processing Systems, NIPS, 2020.
https://proceedings.neurips.cc/paper/2020/file/24389bfe4fe2eba8bf9aa9203a44cdad-Paper.pdf
"""

import os
import warnings

warnings.filterwarnings('ignore')

os.environ[
    'config_file'] = 'examples/multistep_ml_pfl/ms_ml_pfl_fashion_lenet5.yml'

import ms_ml_pfl_server
import ms_ml_pfl_client
import ms_ml_pfl_trainer

from fashion_mnist import DataSource


def main():
    """ A Plato federated learning training session using the one-step MAML algorithm. """

    fashionmnist_datasource = DataSource()

    trainer = ms_ml_pfl_trainer.Trainer()
    client = ms_ml_pfl_client.Client(trainer=trainer,
                                     datasource=fashionmnist_datasource)
    server = ms_ml_pfl_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
