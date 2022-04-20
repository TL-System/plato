"""
Reading runtime parameters from a standard configuration file (which is easier
to work on than JSON).
"""
import argparse
import json
import logging
import os
import sqlite3
from collections import OrderedDict, namedtuple
from typing import Any, IO

import numpy as np
import yaml

from plato.utils.available_gpu import available_gpu


class Loader(yaml.SafeLoader):
    """ YAML Loader with `!include` constructor. """

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self.root_path = os.path.split(stream.name)[0]
        except AttributeError:
            self.root_path = os.path.curdir

        super().__init__(stream)


class Config:
    """
    Retrieving configuration parameters by parsing a configuration file
    using the YAML configuration file parser.
    """

    _instance = None

    @staticmethod
    def construct_include(loader: Loader, node: yaml.Node) -> Any:
        """Include file referenced at node."""

        filename = os.path.abspath(
            os.path.join(loader.root_path, loader.construct_scalar(node)))
        extension = os.path.splitext(filename)[1].lstrip('.')

        with open(filename, 'r', encoding='utf-8') as config_file:
            if extension in ('yaml', 'yml'):
                return yaml.load(config_file, Loader)
            elif extension in ('json', ):
                return json.load(config_file)
            else:
                return ''.join(config_file.readlines())

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
            parser.add_argument(
                '-r',
                '--resume',
                action='store_true',
                help="Resume a previously interrupted training session.")
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

            numeric_level = getattr(logging, args.log.upper(), None)

            if not isinstance(numeric_level, int):
                raise ValueError(f'Invalid log level: {args.log}')

            logging.basicConfig(
                format='[%(levelname)s][%(asctime)s]: %(message)s',
                datefmt='%H:%M:%S')
            root_logger = logging.getLogger()
            root_logger.setLevel(numeric_level)

            cls._instance = super(Config, cls).__new__(cls)

            if 'config_file' in os.environ:
                filename = os.environ['config_file']
            else:
                filename = args.config

            yaml.add_constructor('!include', Config.construct_include, Loader)

            if os.path.isfile(filename):
                with open(filename, 'r', encoding="utf-8") as config_file:
                    config = yaml.load(config_file, Loader)
            else:
                # if the configuration file does not exist, raise an error
                raise ValueError("A configuration file must be supplied.")

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

            if hasattr(Config.clients,
                       "speed_simulation") and Config.clients.speed_simulation:
                Config.simulate_client_speed()

            # Customizable dictionary of global parameters
            Config.params: dict = {}

            # A run ID is unique to each client in an experiment
            Config.params['run_id'] = os.getpid()

            # Pretrained models
            if hasattr(Config().server, 'model_dir'):
                Config.params['model_dir'] = Config().server.model_dir
            else:
                Config.params['model_dir'] = "./models/pretrained"
            os.makedirs(Config.params['model_dir'], exist_ok=True)

            # Resume checkpoint
            if hasattr(Config().server, 'checkpoint_dir'):
                Config.params['checkpoint_dir'] = Config(
                ).server.checkpoint_dir
            else:
                Config.params['checkpoint_dir'] = "./checkpoints"
            os.makedirs(Config.params['checkpoint_dir'], exist_ok=True)

            datasource = Config.data.datasource
            model = Config.trainer.model_name
            server_type = "custom"
            if hasattr(Config().server, "type"):
                server_type = Config.server.type
            elif hasattr(Config().algorithm, "type"):
                server_type = Config.algorithm.type
            Config.params[
                'result_dir'] = f'./results/{datasource}_{model}_{server_type}'

            if 'results' in config:
                Config.results = Config.namedtuple_from_dict(config['results'])

                if hasattr(Config.results, 'result_dir'):
                    Config.params['result_dir'] = Config.results.result_dir

            os.makedirs(Config.params['result_dir'], exist_ok=True)

            if 'model' in config:
                Config.model = Config.namedtuple_from_dict(config['model'])

            if hasattr(Config().trainer, 'max_concurrency'):
                # Using a temporary SQLite database to limit the maximum number of concurrent
                # trainers
                Config.sql_connection = sqlite3.connect(
                    f"{Config.params['result_dir']}/running_trainers.sqlitedb")
                Config().cursor = Config.sql_connection.cursor()

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
    def simulate_client_speed() -> float:
        """Randomly generate a sleep time (in seconds per epoch) for each of the clients."""
        # a random seed must be supplied to make sure that all the clients generate
        # the same set of sleep times per epoch across the board
        if hasattr(Config.clients, "random_seed"):
            np.random.seed(Config.clients.random_seed)
        else:
            np.random.seed(1)

        # Limit the simulated sleep time by the threshold 'max_sleep_time'
        max_sleep_time = 60
        if hasattr(Config.clients, "max_sleep_time"):
            max_sleep_time = Config.clients.max_sleep_time

        dist = Config.clients.simulation_distribution
        total_clients = Config.clients.total_clients
        sleep_times = []

        if hasattr(Config.clients, "simulation_distribution"):

            if dist.distribution.lower() == "normal":
                sleep_times = np.random.normal(dist.mean,
                                               dist.sd,
                                               size=total_clients)
            if dist.distribution.lower() == "pareto":
                sleep_times = np.random.pareto(dist.alpha, size=total_clients)
            if dist.distribution.lower() == "zipf":
                sleep_times = np.random.zipf(dist.s, size=total_clients)
            if dist.distribution.lower() == "uniform":
                sleep_times = np.random.uniform(dist.low,
                                                dist.high,
                                                size=total_clients)
        else:
            # By default, use Pareto distribution with a parameter of 1.0
            sleep_times = np.random.pareto(1.0, size=total_clients)

        Config.client_sleep_times = np.minimum(
            sleep_times, np.repeat(max_sleep_time, total_clients))

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
                gpu_id = int(os.getenv('GPU_ID'))

                if gpu_id is None:
                    gpu_id = available_gpu()
                    os.environ['GPU_ID'] = str(gpu_id)

                tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')

        else:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                if hasattr(Config().trainer,
                           'parallelized') and Config().trainer.parallelized:
                    device = 'cuda'
                else:
                    gpu_id = os.getenv('GPU_ID')

                    if gpu_id is None:
                        gpu_id = available_gpu()
                        os.environ['GPU_ID'] = str(gpu_id)

                    device = f'cuda:{gpu_id}'

        return device

    @staticmethod
    def is_parallel() -> bool:
        """Check if the hardware and OS support data parallelism."""
        import torch

        return hasattr(Config().trainer, 'parallelized') and Config(
        ).trainer.parallelized and torch.cuda.is_available(
        ) and torch.distributed.is_available(
        ) and torch.cuda.device_count() > 1
