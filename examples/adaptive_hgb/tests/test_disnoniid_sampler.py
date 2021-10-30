"""
Test the distribution-based lable nonIID

"""

import os

import torch
from plato.datasources.cifar10 import DataSource
from plato.samplers import registry as samplers_registry

os.environ['config_file'] = 'configs/TestConfigs/dis_noniid_sampler_test.yml'

if __name__ == "__main__":
    cifar10_datasource = DataSource()

    dis_noniid_sampler = samplers_registry.get(datasource=cifar10_datasource,
                                               client_id=0)

    print("sampled size: ", dis_noniid_sampler.trainset_size())
    print("sampled distribution: ",
          dis_noniid_sampler.get_trainset_condition())

    trainset = cifar10_datasource.get_train_set()
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        shuffle=False,
        batch_size=5,
        sampler=dis_noniid_sampler.get())

    NUM_SOW = 2
    SHOW_ID = 0
    for examples, labels in train_loader:

        examples = examples.view(len(examples), -1)
        print("labels: ", labels)

        if SHOW_ID > NUM_SOW:
            break

        SHOW_ID += 1
