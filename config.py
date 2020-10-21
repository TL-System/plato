from collections import namedtuple
import configparser


class Config(object):
    """ 
    Retrieving configuration parameters by parsing a configuration file 
    using the standard Python config parser. 
    """

    def __init__(self, config):
        self.config = configparser.ConfigParser()
        self.config.read(config)

        self.extract()

    def __extract_section(self, section, fields, defaults):
        config = self.config
        params = []

        for i, field in enumerate(fields):
            if type(defaults[i]) is int:
                params.append(config[section].getint(field, defaults[i]))
            elif type(defaults[i]) is float:
                params.append(config[section].getfloat(field, defaults[i]))
            elif type(defaults[i]) is bool:
                params.append(config[section].getboolean(field, defaults[i]))
            else: # the parameter is a string
                params.append(config[section].get(field, defaults[i]))
        
        return params


    def extract(self):
        ''' Extract the parameters from a configuration file. '''

        # Parameters for the federated learning clients
        fields = ['total', 'per_round']
        defaults = (0, 0, 'uniform')
        params = self.__extract_section('clients', fields, defaults)
        self.clients = namedtuple('clients', fields)(*params)

        assert self.clients.per_round <= self.clients.total

        # Parameters for the data distribution
        fields = ['loading', 'partition', 'IID', 'bias', 'shard']
        defaults = ('static', 0, True, None, None)
        params = self.__extract_section('data', fields, defaults)
        self.data = namedtuple('data', fields)(*params)

        # Determine the correct data loader
        assert self.data.IID ^ bool(self.data.bias) ^ bool(self.data.shard)
        if self.data.IID:
            self.loader = 'basic'
        elif self.data.bias:
            self.loader = 'bias'
        elif self.data.shard:
            self.loader = 'shard'

        # Parameters in general for federated learning
        fields = ['rounds', 'target_accuracy', 'task', 'epochs', 'batch_size', 
        'model', 'data_path', 'model_path', 'server']
        defaults = (0, None, 'train', 0, 0, 'MNIST', './data', './models', 'basic')
        params = self.__extract_section('general', fields, defaults)

        self.general = namedtuple('general', fields)(*params)