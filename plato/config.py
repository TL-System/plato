"""
Reading runtime parameters from a standard configuration file (which is easier
to work on than JSON).
"""

import argparse
import json
import logging
import os
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import IO, Any

import numpy as np
import yaml


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

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
        with open(
            Path(loader.name)
            .parent.joinpath(loader.construct_yaml_str(node))
            .resolve(),
            "r",
        ) as f:
            return yaml.load(f, type(loader))

    def __new__(cls):
        if cls._instance is None:
            parser = argparse.ArgumentParser()
            parser.add_argument("-i", "--id", type=str, help="Unique client ID.")
            parser.add_argument(
                "-p", "--port", type=str, help="The port number for running a server."
            )
            parser.add_argument(
                "-c",
                "--config",
                type=str,
                default="./config.yml",
                help="Federated learning configuration file.",
            )
            parser.add_argument(
                "-b",
                "--base",
                type=str,
                default="./",
                help="The base path for datasets and models.",
            )
            parser.add_argument(
                "-s",
                "--server",
                type=str,
                default=None,
                help="The server hostname and port number.",
            )
            parser.add_argument(
                "-u", "--cpu", action="store_true", help="Use CPU as the device."
            )
            parser.add_argument(
                "-m", "--mps", action="store_true", help="Use MPS as the device."
            )
            parser.add_argument(
                "-d",
                "--download",
                action="store_true",
                help="Download the dataset to prepare for a training session.",
            )
            parser.add_argument(
                "-r",
                "--resume",
                action="store_true",
                help="Resume a previously interrupted training session.",
            )
            parser.add_argument(
                "-l", "--log", type=str, default="info", help="Log messages level."
            )

            args = parser.parse_args()
            Config.args = args

            if Config.args.id is not None:
                Config.args.id = int(args.id)
            if Config.args.port is not None:
                Config.args.port = int(args.port)

            numeric_level = getattr(logging, args.log.upper(), None)

            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level: {args.log}")

            logging.basicConfig(
                format="[%(levelname)s][%(asctime)s]: %(message)s", datefmt="%H:%M:%S"
            )

            root_logger = logging.getLogger()
            root_logger.setLevel(numeric_level)

            cls._instance = super(Config, cls).__new__(cls)

            if "config_file" in os.environ:
                filename = os.environ["config_file"]
            else:
                filename = args.config

            yaml.add_constructor("!include", Config.construct_include, Loader)

            if os.path.isfile(filename):
                with open(filename, "r", encoding="utf-8") as config_file:
                    config = yaml.load(config_file, Loader)
            else:
                # if the configuration file does not exist, raise an error
                raise ValueError("A configuration file must be supplied.")

            Config.clients = Config.namedtuple_from_dict(config["clients"])
            Config.server = Config.namedtuple_from_dict(config["server"])
            Config.data = Config.namedtuple_from_dict(config["data"])
            Config.trainer = Config.namedtuple_from_dict(config["trainer"])
            Config.algorithm = Config.namedtuple_from_dict(config["algorithm"])

            if Config.args.server is not None:
                Config.server = Config.server._replace(
                    address=args.server.split(":")[0]
                )
                Config.server = Config.server._replace(port=args.server.split(":")[1])

            if Config.args.download:
                Config.clients = Config.clients._replace(total_clients=1)
                Config.clients = Config.clients._replace(per_round=1)

            if (
                hasattr(Config.clients, "speed_simulation")
                and Config.clients.speed_simulation
            ):
                Config.simulate_client_speed()

            # Customizable dictionary of global parameters
            Config.params: dict = {}

            # A run ID is unique to each client in an experiment
            Config.params["run_id"] = os.getpid()

            # The base path used for all datasets, models, checkpoints, and results
            Config.params["base_path"] = Config.args.base

            if "general" in config:
                Config.general = Config.namedtuple_from_dict(config["general"])

                if hasattr(Config.general, "base_path"):
                    Config.params["base_path"] = Config().general.base_path

            # Directory of dataset
            if hasattr(Config().data, "data_path"):
                Config.params["data_path"] = os.path.join(
                    Config.params["base_path"], Config().data.data_path
                )
            else:
                Config.params["data_path"] = os.path.join(
                    Config.params["base_path"], "data"
                )

            # Pretrained models
            if hasattr(Config().server, "model_path"):
                Config.params["model_path"] = os.path.join(
                    Config.params["base_path"], Config().server.model_path
                )
            else:
                Config.params["model_path"] = os.path.join(
                    Config.params["base_path"], "models/pretrained"
                )
            os.makedirs(Config.params["model_path"], exist_ok=True)

            # Resume checkpoint
            if hasattr(Config().server, "checkpoint_path"):
                Config.params["checkpoint_path"] = os.path.join(
                    Config.params["base_path"], Config().server.checkpoint_path
                )
            else:
                Config.params["checkpoint_path"] = os.path.join(
                    Config.params["base_path"], "checkpoints"
                )
            os.makedirs(Config.params["checkpoint_path"], exist_ok=True)

            if hasattr(Config().server, "mpc_data_path"):
                mpc_dir = os.path.join(
                    Config.params["base_path"], Config().server.mpc_data_path
                )
            else:
                mpc_dir = os.path.join(Config.params["base_path"], "mpc_data")
            Config.params["mpc_data_path"] = mpc_dir
            os.makedirs(mpc_dir, exist_ok=True)

            if "results" in config:
                Config.results = Config.namedtuple_from_dict(config["results"])

            # Directory of the .csv file containing results
            if hasattr(Config, "results") and hasattr(Config.results, "result_path"):
                Config.params["result_path"] = os.path.join(
                    Config.params["base_path"], Config.results.result_path
                )
            else:
                Config.params["result_path"] = os.path.join(
                    Config.params["base_path"], "results"
                )
            os.makedirs(Config.params["result_path"], exist_ok=True)

            # The set of columns in the .csv file
            if hasattr(Config, "results") and hasattr(Config.results, "types"):
                Config.params["result_types"] = Config.results.types
            else:
                Config.params["result_types"] = "round, accuracy, elapsed_time"

            # The set of pairs to be plotted
            if hasattr(Config, "results") and hasattr(Config.results, "plot"):
                Config.params["plot_pairs"] = Config().results.plot
            else:
                Config.params["plot_pairs"] = "round-accuracy, elapsed_time-accuracy"

            if "parameters" in config:
                Config.parameters = Config.namedtuple_from_dict(config["parameters"])

        return cls._instance

    @staticmethod
    def namedtuple_from_dict(obj):
        """Creates a named tuple from a dictionary."""
        if isinstance(obj, dict):
            fields = sorted(obj.keys())
            namedtuple_type = namedtuple(
                typename="Config", field_names=fields, rename=True
            )
            field_value_pairs = OrderedDict(
                (str(field), Config.namedtuple_from_dict(obj[field]))
                for field in fields
            )
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
                sleep_times = np.random.normal(dist.mean, dist.sd, size=total_clients)
            if dist.distribution.lower() == "pareto":
                sleep_times = np.random.pareto(dist.alpha, size=total_clients)
            if dist.distribution.lower() == "zipf":
                sleep_times = np.random.zipf(dist.s, size=total_clients)
            if dist.distribution.lower() == "uniform":
                sleep_times = np.random.uniform(dist.low, dist.high, size=total_clients)
        else:
            # By default, use Pareto distribution with a parameter of 1.0
            sleep_times = np.random.pareto(1.0, size=total_clients)

        Config.client_sleep_times = np.minimum(
            sleep_times, np.repeat(max_sleep_time, total_clients)
        )

    @staticmethod
    def is_edge_server() -> bool:
        """Returns whether the current instance is an edge server in cross-silo FL."""
        return Config().args.port is not None

    @staticmethod
    def is_central_server() -> bool:
        """Returns whether the current instance is a central server in cross-silo FL."""
        return hasattr(Config().algorithm, "cross_silo") and Config().args.port is None

    @staticmethod
    def gpu_count() -> int:
        """Returns the number of GPUs available for training."""

        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
        elif Config.args.mps and torch.backends.mps.is_built():
            return 1
        else:
            return 0

    @staticmethod
    def device() -> str:
        """Returns the device to be used for training."""
        device = "cpu"

        if Config.args.cpu:
            return device

        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            if Config.gpu_count() > 1 and isinstance(Config.args.id, int):
                # A client will always run on the same GPU
                gpu_id = Config.args.id % torch.cuda.device_count()
                device = f"cuda:{gpu_id}"
            else:
                device = "cuda:0"

        if Config.args.mps and torch.backends.mps.is_built():
            device = "mps"

        return device
