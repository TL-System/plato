"""
Reading runtime parameters from a standard configuration file (which is easier
to work on than JSON).
"""
import argparse
import logging
import os
import random
import sqlite3
from collections import OrderedDict, namedtuple

import yaml
from yamlinclude import YamlIncludeConstructor


class Config:
    """
    Retrieving configuration parameters by parsing a configuration file
    using the YAML configuration file parser.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            parser = argparse.ArgumentParser()
            parser.add_argument('-i',
                                '--id',
                                type=str,
                                help='Unique client ID.')
            parser.add_argument('-p',
                                '--port',
                                type=str,
                                help='The port number for running a server.')
            parser.add_argument('-c',
                                '--config',
                                type=str,
                                default='./config.yml',
                                help='Federated learning configuration file.')
            parser.add_argument('-s',
                                '--server',
                                type=str,
                                default=None,
                                help='The server hostname and port number.')
            parser.add_argument(
                '-d',
                '--download',
                action='store_true',
                help='Download the dataset to prepare for a training session.')
            parser.add_argument('-l',
                                '--log',
                                type=str,
                                default='info',
                                help='Log messages level.')

            args = parser.parse_args()
            Config.args = args

            if Config.args.id is not None:
                Config.args.id = int(args.id)
            if Config.args.port is not None:
                Config.args.port = int(args.port)

            try:
                log_level = {
                    'critical': logging.CRITICAL,
                    'error': logging.ERROR,
                    'warn': logging.WARN,
                    'info': logging.INFO,
                    'debug': logging.DEBUG
                }[args.log]
            except KeyError:
                log_level = logging.INFO

            logging.basicConfig(
                format='[%(levelname)s][%(asctime)s]: %(message)s',
                level=log_level,
                datefmt='%H:%M:%S')

            cls._instance = super(Config, cls).__new__(cls)

            if 'config_file' in os.environ:
                filename = os.environ['config_file']
            else:
                filename = args.config

            YamlIncludeConstructor.add_to_loader_class(
                loader_class=yaml.SafeLoader, base_dir='./configs')

            if os.path.isfile(filename):
                with open(filename, 'r', encoding="utf8") as config_file:
                    config = yaml.load(config_file, Loader=yaml.SafeLoader)
            else:
                # if the configuration file does not exist, use a default one
                config = Config.default_config()

            Config.clients = Config.namedtuple_from_dict(config['clients'])
            Config.server = Config.namedtuple_from_dict(config['server'])
            Config.data = Config.namedtuple_from_dict(config['data'])
            Config.trainer = Config.namedtuple_from_dict(config['trainer'])
            Config.algorithm = Config.namedtuple_from_dict(config['algorithm'])

            if Config.args.server is not None:
                Config.server = Config.server._replace(
                    address=args.server.split(':')[0])
                Config.server = Config.server._replace(
                    port=args.server.split(':')[1])

            if Config.args.download:
                Config.clients = Config.clients._replace(total_clients=1)
                Config.clients = Config.clients._replace(per_round=1)

            if 'results' in config:
                Config.results = Config.namedtuple_from_dict(config['results'])
                if hasattr(Config().results, 'results_dir'):
                    Config.result_dir = Config.results.results_dir
                else:
                    datasource = Config.data.datasource
                    model = Config.trainer.model_name
                    server_type = Config.algorithm.type
                    Config.result_dir = f'./results/{datasource}/{model}/{server_type}/'

            if 'model' in config:
                Config.model = Config.namedtuple_from_dict(config['model'])

            if hasattr(Config().trainer, 'max_concurrency'):
                # Using a temporary SQLite database to limit the maximum number of concurrent
                # trainers
                Config.sql_connection = sqlite3.connect(
                    "/tmp/running_trainers.sqlitedb")
                Config().cursor = Config.sql_connection.cursor()

            # Customizable dictionary of global parameters
            Config.params: dict = {}

            # A run ID is unique to each client in an experiment
            Config.params['run_id'] = os.getpid()

            # Pretrained models
            Config.params['model_dir'] = "./models/pretrained/"
            Config.params['pretrained_model_dir'] = "./models/pretrained/"

        return cls._instance

    @staticmethod
    def namedtuple_from_dict(obj):
        """Creates a named tuple from a dictionary."""
        if isinstance(obj, dict):
            fields = sorted(obj.keys())
            namedtuple_type = namedtuple(typename='Config',
                                         field_names=fields,
                                         rename=True)
            field_value_pairs = OrderedDict(
                (str(field), Config.namedtuple_from_dict(obj[field]))
                for field in fields)
            try:
                return namedtuple_type(**field_value_pairs)
            except TypeError:
                # Cannot create namedtuple instance so fallback to dict (invalid attribute names)
                return dict(**field_value_pairs)
        elif isinstance(obj, (list, set, tuple, frozenset)):
            return [Config.namedtuple_from_dict(item) for item in obj]
        else:
            return obj

    @staticmethod
    def is_edge_server() -> bool:
        """Returns whether the current instance is an edge server in cross-silo FL."""
        return Config().args.port is not None

    @staticmethod
    def is_central_server() -> bool:
        """Returns whether the current instance is a central server in cross-silo FL."""
        return hasattr(Config().algorithm,
                       'cross_silo') and Config().args.port is None

    @staticmethod
    def device() -> str:
        """Returns the device to be used for training."""
        device = 'cpu'
        if hasattr(Config().trainer, 'use_mindspore'):
            pass
        elif hasattr(Config().trainer, 'use_tensorflow'):
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if len(gpus) > 0:
                device = 'GPU'
                tf.config.experimental.set_visible_devices(
                    gpus[random.randint(0,
                                        len(gpus) - 1)], 'GPU')
        else:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                if hasattr(Config().trainer,
                           'parallelized') and Config().trainer.parallelized:
                    device = 'cuda'
                else:
                    device = 'cuda:' + str(
                        random.randint(0,
                                       torch.cuda.device_count() - 1))

        return device

    @staticmethod
    def is_parallel() -> bool:
        """Check if the hardware and OS support data parallelism."""
        import torch

        return hasattr(Config().trainer, 'parallelized') and Config(
        ).trainer.parallelized and torch.cuda.is_available(
        ) and torch.distributed.is_available(
        ) and torch.cuda.device_count() > 1

    @staticmethod
    def default_config() -> dict:
        ''' Supply a default configuration when the configuration file is missing. '''
        config = {}
        config['clients'] = {}
        config['clients']['type'] = 'simple'
        config['clients']['total_clients'] = 1
        config['clients']['per_round'] = 1
        config['clients']['do_test'] = False
        config['server'] = {}
        config['server']['address'] = '127.0.0.1'
        config['server']['port'] = 8000
        config['server']['disable_clients'] = True
        config['data'] = {}
        config['data']['datasource'] = 'MNIST'
        config['data']['data_path'] = './data'
        config['data']['partition_size'] = 20000
        config['data']['sampler'] = 'iid'
        config['data']['random_seed'] = 1
        config['trainer'] = {}
        config['trainer']['type'] = 'basic'
        config['trainer']['rounds'] = 5
        config['trainer']['parallelized'] = False
        config['trainer']['target_accuracy'] = 0.94
        config['trainer']['epochs'] = 5
        config['trainer']['batch_size'] = 32
        config['trainer']['optimizer'] = 'SGD'
        config['trainer']['learning_rate'] = 0.01
        config['trainer']['momentum'] = 0.9
        config['trainer']['weight_decay'] = 0.0
        config['trainer']['model_name'] = 'lenet5'
        config['algorithm'] = {}
        config['algorithm']['type'] = 'fedavg'

        return config

    @staticmethod
    def store() -> None:
        """ Saving the current run-time configuration to a file. """
        data = {}
        data['clients'] = Config.clients._asdict()
        data['server'] = Config.server._asdict()
        data['data'] = Config.data._asdict()
        data['trainer'] = Config.trainer._asdict()
        data['algorithm'] = Config.algorithm._asdict()
        with open(Config.args.config, "w", encoding="utf8") as out:
            yaml.dump(data, out, default_flow_style=False)
