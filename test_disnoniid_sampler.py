#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

os.environ['config_file'] = 'configs/TestConfigs/dis_noniid_sampler_test.yml'

import torch
from plato.datasources.cifar10 import DataSource
from plato.samplers.distribution_noniid import Sampler

if __name__ == "__main__":
    cifar10_datasource = DataSource()
    dis_noniid_sampler = Sampler(datasource=cifar10_datasource, client_id=1)

    trainset = cifar10_datasource.get_train_set()
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        shuffle=False,
        batch_size=5,
        sampler=dis_noniid_sampler.get())

    print("train_loader: ", train_loader)
    num_sow = 2
    show_id = 0
    for examples, labels in train_loader:

        examples = examples.view(len(examples), -1)
        print("examples: ", examples)
        print("labels: ", labels)

        if show_id > num_sow:
            break

        show_id += 1