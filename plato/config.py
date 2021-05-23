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

            with open(filename, 'r') as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)

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

            if 'results' in config:
                Config.results = Config.namedtuple_from_dict(config['results'])
                Config.result_dir = os.path.dirname(__file__) + '/results/'

            # Used to limit the maximum number of concurrent trainers
            Config.sql_connection = sqlite3.connect(
                os.path.dirname(__file__) + '/running_trainers.sqlitedb')

            Config().cursor = Config.sql_connection.cursor()

            # Customizable dictionary of global parameters
            Config.params: dict = {}

            # A run ID is unique to each client in an experiment
            Config.params['run_id'] = os.getpid()

            # Pretrained models
            Config.params['model_dir'] = "./models/pretrained/"

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
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            if hasattr(Config().trainer,
                       'parallelized') and Config().trainer.parallelized:
                device = 'cuda'
            else:
                device = 'cuda:' + str(
                    random.randint(0,
                                   torch.cuda.device_count() - 1))
        else:
            device = 'cpu'

        return device

    @staticmethod
    def is_parallel() -> bool:
        """Check if the hardware and OS support data parallelism."""
        import torch

        return hasattr(Config().trainer, 'parallelized') and Config(
        ).trainer.parallelized and torch.cuda.is_available(
        ) and torch.distributed.is_available(
        ) and torch.cuda.device_count() > 1
