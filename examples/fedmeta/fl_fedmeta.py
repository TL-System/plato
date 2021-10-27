"""
Improving federated learning using MAML algorithm

Reference:
Chen, Fei and Luo, Mi and Dong, Zhenhua and Li, Zhenguo and He, Xiuqiang.
"Federated Meta-Learning with Fast Convergence and Efficient Communication"
arXiv preprint arXiv:1802.07876, 2018.
https://arxiv.org/pdf/1802.07876.pdf
"""

import os

os.environ['config_file'] = './fl_meta_MNIST_lenet5.yml'

import fedmeta_server
import fedmeta_client
import fedmeta_trainer


def main():
    """ A Plato federated learning training session using the MAML algorithm. """
    trainer = fedmeta_trainer.Trainer()
    client = fedmeta_client.Client(trainer=trainer)
    server = fedmeta_server.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
