

import os
import sys

os.environ['config_file'] = 'configs/TestConfigs/dis_noniid_sampler_test.yml'

import torch
from plato.datasources.cifar10 import DataSource
from plato.samplers import modality_registry

if __name__ == "__main__":
    cifar10_datasource = DataSource()
    defined_sampler = modality_registry
    dis_noniid_sampler = Sampler(datasource=cifar10_datasource, client_id=0)

    print("sampled size: ", dis_noniid_sampler.trainset_size())
    print("sampled distribution: ",
          dis_noniid_sampler.get_trainset_condition())

    trainset = cifar10_datasource.get_train_set()
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        shuffle=False,
        batch_size=5,
        sampler=dis_noniid_sampler.get())

    num_sow = 2
    show_id = 0
    for examples, labels in train_loader:

        examples = examples.view(len(examples), -1)
        print("labels: ", labels)

        if show_id > num_sow:
            break

        show_id += 1