"""
Reading runtime parameters from a standard configuration file (which is easier
to work on than JSON).
"""

from collections import namedtuple
import configparser


class Config:
    """
    Retrieving configuration parameters by parsing a configuration file
    using the standard Python config parser.
    """

    def __init__(self, config):
        self.config = configparser.ConfigParser()
        self.config.read(config)

        self.extract()

    def __extract_section(self, section, fields, defaults):
        """Extract the parameters from one section in a configuration file."""
        config = self.config
        params = []

        for i, field in enumerate(fields):
            if type(defaults[i]) is int:
                params.append(config[section].getint(field, defaults[i]))
            elif type(defaults[i]) is float:
                params.append(config[section].getfloat(field, defaults[i]))
            elif type(defaults[i]) is bool:
                params.append(config[section].getboolean(field, defaults[i]))
            else: # assuming that the parameter is a string
                params.append(config[section].get(field, defaults[i]))

        return params


    def extract(self):
        """Extract the parameters from a configuration file."""

        # Parameters for the federated learning clients
        fields = ['total', 'per_round', 'do_test', 'test_partition']
        defaults = (0, 0, False, 0.2)
        params = self.__extract_section('clients', fields, defaults)
        self.clients = namedtuple('clients', fields)(*params)

        assert self.clients.per_round <= self.clients.total

        # Parameters for the data distribution
        fields = ['partition_size', 'divider', 'label_distribution', 
                  'bias_primary_percentage', 'bias_secondary_focus', 'shard_per_client']
        defaults = (0, 'iid', 'uniform', 0.8, False, 2)
        params = self.__extract_section('data', fields, defaults)
        self.data = namedtuple('data', fields)(*params)

        # Training parameters for federated learning
        fields = ['rounds', 'target_accuracy', 'task', 'epochs', 'batch_size', 'dataset',
                  'data_path', 'num_layers', 'num_classes', 'model',
                  'optimizer', 'learning_rate', 'momentum', 'server']
        defaults = (0, 0.9, 'train', 0, 0, 'MNIST', './data', 40, 10, 'mnist_cnn',
                    'SGD', 0.01, 0.5, 'fedavg')
        params = self.__extract_section('training', fields, defaults)

        self.training = namedtuple('training', fields)(*params)

        # Parameters for the federated learning server
        fields = ['address', 'port']
        defaults = ('localhost', 8000)
        params = self.__extract_section('server', fields, defaults)

        self.server = namedtuple('server', fields)(*params)
