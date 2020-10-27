"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

from datasets import mnist, fashion_mnist, cifar10

registered_datasets = {'MNIST': mnist,
                       'FashionMNIST': fashion_mnist,
                       'CIFAR10': cifar10
                      }


def get(dataset_name):
    """Get the dataset with the provided name."""
    if dataset_name in registered_datasets:
        dataset = registered_datasets[dataset_name].Dataset()
    else:
        raise ValueError('No such dataset: {}'.format(dataset_name))

    return dataset
