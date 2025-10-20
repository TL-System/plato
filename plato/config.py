"""
Reading runtime parameters from a standard configuration file (which is easier
to work on than JSON).
"""

from __future__ import annotations

import argparse
import logging
import os
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from munch import Munch


class ConfigNode(Munch):
    """Dictionary-like container with dot access and `_replace` compatibility."""

    @classmethod
    def from_object(cls, obj: Any) -> Any:
        """Recursively convert mappings into ``ConfigNode`` instances."""
        if isinstance(obj, dict):
            return cls({key: cls.from_object(value) for key, value in obj.items()})
        if isinstance(obj, list):
            return [cls.from_object(item) for item in obj]
        return obj

    def _replace(self, **updates: Any) -> "ConfigNode":
        """Return a new instance with the provided fields updated."""
        data = dict(self)
        for key, value in updates.items():
            data[key] = self.from_object(value)
        return type(self).from_object(data)

    def _asdict(self) -> dict[str, Any]:
        """Return a plain dictionary representation of the node."""
        return {key: self._to_plain(value) for key, value in self.items()}

    @classmethod
    def _to_plain(cls, value: Any) -> Any:
        if isinstance(value, ConfigNode):
            return value._asdict()
        if isinstance(value, list):
            return [cls._to_plain(item) for item in value]
        return value


class TomlConfigLoader:
    """Load TOML configuration files with simple include semantics."""

    def __init__(self, root_path: Path) -> None:
        self._root_path = Path(root_path)

    def load(self) -> Any:
        """Load and resolve the configuration rooted at ``root_path``."""
        return self._load_file(self._root_path.resolve(), set())

    def _load_file(self, filename: Path, seen: set[Path]) -> Any:
        if filename in seen:
            raise ValueError(f"Circular include detected while loading {filename}.")
        seen.add(filename)
        with filename.open("rb") as handle:
            data = tomllib.load(handle)
        return self._resolve(data, filename.parent, seen)

    def _resolve(self, value: Any, base_dir: Path, seen: set[Path]) -> Any:
        if isinstance(value, dict):
            if set(value.keys()) == {"null"} and value["null"] is True:
                return None
            if "include" in value:
                include_spec = value["include"]
                overrides = {k: v for k, v in value.items() if k != "include"}
                included = self._resolve_include(include_spec, base_dir, seen)
                resolved_overrides = (
                    self._resolve(overrides, base_dir, seen) if overrides else {}
                )
                return self._merge(included, resolved_overrides)
            return {
                key: self._resolve(item, base_dir, seen) for key, item in value.items()
            }
        if isinstance(value, list):
            resolved_list = [self._resolve(item, base_dir, seen) for item in value]
            if resolved_list and all(
                item is None
                or (isinstance(item, dict) and set(item.keys()) == {"value"})
                for item in resolved_list
            ):
                normalized_list = []
                for item in resolved_list:
                    if item is None:
                        normalized_list.append(None)
                    else:
                        normalized_list.append(item["value"])
                return normalized_list
            return resolved_list
        return value

    def _resolve_include(
        self, include_spec: Any, base_dir: Path, seen: set[Path]
    ) -> Any:
        if isinstance(include_spec, str):
            include_path = self._resolve_path(include_spec, base_dir)
            return self._load_file(include_path, seen)
        if isinstance(include_spec, list):
            aggregated: Any = None
            for entry in include_spec:
                included = self._resolve_include(entry, base_dir, seen)
                aggregated = (
                    included
                    if aggregated is None
                    else self._merge(aggregated, included)
                )
            return aggregated
        raise TypeError("Include directive must be a string or list of strings.")

    @staticmethod
    def _merge(base: Any, override: Any) -> Any:
        if base is None:
            return override
        if override is None:
            return base
        if isinstance(base, dict) and isinstance(override, dict):
            merged = dict(base)
            for key, value in override.items():
                merged[key] = TomlConfigLoader._merge(merged.get(key), value)
            return merged
        if isinstance(base, list) and isinstance(override, list):
            return base + override
        return override

    @staticmethod
    def _resolve_path(candidate: str, base_dir: Path) -> Path:
        path = (base_dir / candidate).resolve()
        return path


class Config:
    """
    Retrieving configuration parameters by parsing a configuration file
    using the TOML configuration file parser.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            parser = argparse.ArgumentParser()
            if not parser.prog.startswith("uv run "):
                parser.prog = f"uv run {parser.prog}"
            parser.add_argument("-i", "--id", type=str, help="Unique client ID.")
            parser.add_argument(
                "-p", "--port", type=str, help="The port number for running a server."
            )
            parser.add_argument(
                "-c",
                "--config",
                type=str,
                default="./config.toml",
                help="Federated learning configuration file.",
            )
            parser.add_argument(
                "-b",
                "--base",
                type=str,
                default="./runtime",
                help="The base path for datasets, models, checkpoints, and results.",
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

            config_path = Path(filename)
            if config_path.is_file():
                loader = TomlConfigLoader(config_path)
                raw_config = loader.load()
                config = ConfigNode.from_object(raw_config)
            else:
                usage = parser.format_usage().strip()
                raise SystemExit(
                    "Please provide a configuration file using the '-c' option.\n"
                    f"{usage}"
                )

            Config.config_path = config_path
            Config.config = config
            Config.clients = config.clients
            Config.server = config.server
            Config.data = config.data
            Config.trainer = config.trainer
            Config.algorithm = config.algorithm

            if Config.args.server is not None:
                address, port = args.server.split(":")
                Config.server.address = address
                Config.server.port = int(port)

            if (
                hasattr(Config.clients, "speed_simulation")
                and Config.clients.speed_simulation
            ):
                Config.simulate_client_speed()

            # Customizable dictionary of global parameters
            Config.params: dict[str, Any] = {}

            # A run ID is unique to each client in an experiment
            Config.params["run_id"] = os.getpid()

            # The base path used for all datasets, models, checkpoints, and results
            Config.params["base_path"] = Config.args.base

            if hasattr(config, "general"):
                Config.general = config.general

                if hasattr(Config.general, "base_path"):
                    Config.params["base_path"] = Config().general.base_path

            os.makedirs(Config.params["base_path"], exist_ok=True)

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

            if hasattr(config, "results"):
                Config.results = config.results

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

            if hasattr(config, "parameters"):
                Config.parameters = config.parameters

        return cls._instance

    @staticmethod
    def node_from_dict(obj: Any) -> Any:
        """Construct a ``ConfigNode`` (recursively) from a plain mapping."""
        return ConfigNode.from_object(obj)

    namedtuple_from_dict = node_from_dict

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
