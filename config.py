"""
Reading runtime parameters from a standard configuration file (which is easier
to work on than JSON).
"""

import logging
from collections import namedtuple, OrderedDict
import os
import sqlite3
import uuid

import argparse
import torch
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
            parser.add_argument('-l',
                                '--log',
                                type=str,
                                default='info',
                                help='Log messages level.')

            args = parser.parse_args()

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

            with open(args.config, 'r') as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)

            Config.clients = Config.namedtuple_from_dict(config['clients'])
            Config.server = Config.namedtuple_from_dict(config['server'])
            Config.data = Config.namedtuple_from_dict(config['data'])
            Config.trainer = Config.namedtuple_from_dict(config['trainer'])
            Config.algorithm = Config.namedtuple_from_dict(config['algorithm'])

            if 'results' in config:
                Config.results = Config.namedtuple_from_dict(config['results'])
                Config.result_dir = os.path.dirname(args.config) + '/results/'

            Config.args = args

            # Used to limit the maximum number of concurrent trainers
            Config.sql_connection = sqlite3.connect(
                './running_trainers.sqlitedb')
            Config().cursor = Config.sql_connection.cursor()

            # Customizable dictionary of global parameters
            Config.params = {}

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
    def is_edge_server():
        """Returns whether the current instance is an edge server in cross-silo FL."""
        return Config().args.port is not None

    @staticmethod
    def is_central_server():
        """Returns whether the current instance is a central server in cross-silo FL."""
        return hasattr(Config().algorithm,
                       'cross_silo') and Config().args.port is None

    @staticmethod
    def device():
        """Returns the device to be used for training."""
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = 'cuda'
        else:
            device = 'cpu'

        return device

    @staticmethod
    def is_parallel():
        """Check if the hardware and OS support data parallelism."""
        return hasattr(Config().trainer, 'parallelized') and Config(
        ).trainer.parallelized and torch.cuda.is_available(
        ) and torch.distributed.is_available(
        ) and torch.cuda.device_count() > 1
