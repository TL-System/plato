"""
A federated learning training session using AFL.

Reference:

Goetz et al., "Active Federated Learning".

https://arxiv.org/pdf/1909.12641.pdf
"""
import os


os.environ['config_file'] = 'examples/afl/afl_MNIST_lenet5.yml'

import numpy as np
import torch
import torch.nn as nn
import wandb

import afl_server
import afl_client
import afl_trainer


def main():
    """ A Plato federated learning training session using the AFL algorithm. """
    trainer = afl_trainer.Trainer()
    client = afl_client.Client(trainer=trainer)
    server = afl_server.Server()
    server.run(client)


if __name__ == "__main__":
    main()
