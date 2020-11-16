"""
Reading runtime parameters from a standard configuration file (which is easier
to work on than JSON).
"""

import logging
from collections import namedtuple
import configparser
import argparse
import torch

class Config:
    """
    Retrieving configuration parameters by parsing a configuration file
    using the standard Python config parser.
    """

    _instance = None
    config = configparser.ConfigParser()

    def __new__(cls):
        if cls._instance is None:
            print('Reading the configuration file...')
            parser = argparse.ArgumentParser()
            parser.add_argument('-i', '--id', type=str,
                                help='Unique client ID.')
            parser.add_argument('-e', '--edgeid', type=str,
                                help='Unique edge server ID.')
            parser.add_argument('-c', '--config', type=str, default='./config.conf',
                                help='Federated learning configuration file.')
            parser.add_argument('-l', '--log', type=str, default='info',
                                help='Log messages level.')

            Config.args = parser.parse_args()

            try:
                log_level = {
                    'critical': logging.CRITICAL,
                    'error': logging.ERROR,
                    'warn': logging.WARN,
                    'info': logging.INFO,
                    'debug': logging.DEBUG
                }[Config.args.log]
            except KeyError:
                log_level = logging.INFO

            logging.basicConfig(
                format='[%(levelname)s][%(asctime)s]: %(message)s',
                level=log_level, datefmt='%H:%M:%S')

            cls._instance = super(Config, cls).__new__(cls)
            cls.config.read(Config.args.config)
            cls.extract()
        return cls._instance


    @staticmethod
    def device():
        """Returns the device to be used for training."""
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device_ids = ','.join([str(x) for x in range(torch.cuda.device_count())])
            device = f'cuda:{device_ids}'
        else:
            device = 'cpu'

        return device


    @staticmethod
    def is_parallel():
        """Check if the hardware supports data parallelism."""
        return torch.cuda.is_available() and torch.cuda.device_count() > 1


    @staticmethod
    def extract_section(section, fields, defaults, optional=False):
        """Extract the parameters from one section in a configuration file."""
        params = []

        if optional and section not in Config.config:
            return None

        for i, field in enumerate(fields):
            if isinstance(defaults[i], bool):
                params.append(Config.config[section].getboolean(field, defaults[i]))
            elif isinstance(defaults[i], int):
                params.append(Config.config[section].getint(field, defaults[i]))
            elif isinstance(defaults[i], float):
                params.append(Config.config[section].getfloat(field, defaults[i]))
            else: # assuming that the parameter is a string
                params.append(Config.config[section].get(field, defaults[i]))

        return params


    @staticmethod
    def extract():
        """Extract the parameters from a configuration file."""

        # Parameters for the federated learning clients
        fields = ['total', 'per_round', 'do_test', 'test_partition']
        defaults = (0, 0, False, 0.2)
        params = Config.extract_section('clients', fields, defaults)
        Config.clients = namedtuple('clients', fields)(*params)

        assert Config.clients.per_round <= Config.clients.total

        # Parameters for the data distribution
        fields = ['partition_size', 'divider', 'label_distribution',
                  'bias_primary_percentage', 'bias_secondary_focus', 'shard_per_client']
        defaults = (0, 'iid', 'uniform', 0.8, False, 2)
        params = Config.extract_section('data', fields, defaults)
        Config.data = namedtuple('data', fields)(*params)

        # Training parameters for federated learning
        fields = ['rounds', 'target_accuracy', 'task', 'epochs', 'batch_size', 'dataset',
                  'data_path', 'num_layers', 'num_classes', 'model',
                  'optimizer', 'learning_rate', 'momentum', 'server']
        defaults = (0, 0.9, 'train', 0, 0, 'MNIST', './data', 40, 10, 'mnist_cnn',
                    'SGD', 0.01, 0.5, 'fedavg')
        params = Config.extract_section('training', fields, defaults)

        Config.training = namedtuple('training', fields)(*params)

        # Parameters for the federated learning server
        fields = ['address', 'port']
        defaults = ('localhost', 8000)
        params = Config.extract_section('server', fields, defaults)
        Config.server = namedtuple('server', fields)(*params)

        # If the topology is hierarchical (cross-silo FL training)
        fields = ['total', 'aggregations', 'do_test']
        defaults = (0, 0, False)
        params = Config.extract_section('edges', fields, defaults, optional=True)
        if params is not None:
            Config.edges = namedtuple('edges', fields)(*params)
        else:
            Config.edges = None
