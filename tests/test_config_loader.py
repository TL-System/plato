"""Tests for the TOML configuration loader and ConfigNode helpers."""

from __future__ import annotations

import sys
from pathlib import Path

from plato.config import Config, ConfigNode, TomlConfigLoader
from plato.utils import toml_writer


def test_toml_loader_resolves_include_and_overrides(tmp_path: Path):
    base_path = tmp_path / "clients_base.toml"
    base_path.write_text('type = "simple"\n', encoding="utf-8")

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[clients]
include = "clients_base.toml"
per_round = 2
""",
        encoding="utf-8",
    )

    loader = TomlConfigLoader(config_path)
    config = loader.load()

    assert config["clients"]["type"] == "simple"
    assert config["clients"]["per_round"] == 2


def test_toml_loader_handles_none_and_mixed_lists(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[runner]
load_from = { null = true }

[[runner.workflow]]
value = "train"

[[runner.workflow]]
value = 1
""",
        encoding="utf-8",
    )

    loader = TomlConfigLoader(config_path)
    config = loader.load()

    assert config["runner"]["load_from"] is None
    assert config["runner"]["workflow"] == ["train", 1]


def test_config_node_replace_and_asdict():
    node = Config.node_from_dict({"clients": {"type": "simple", "per_round": 1}})
    assert isinstance(node, ConfigNode)
    assert node.clients.type == "simple"

    updated = node.clients._replace(per_round=5)
    assert updated.per_round == 5
    assert node.clients.per_round == 1
    assert updated._asdict() == {"per_round": 5, "type": "simple"}


def test_cli_arguments_override_config_values(tmp_path: Path, monkeypatch):
    config_base = tmp_path / "config_base"
    cli_base = tmp_path / "cli_base"
    config_path = tmp_path / "override_config.toml"

    config_data = {
        "clients": {"type": "simple", "total_clients": 1, "per_round": 1},
        "server": {"address": "127.0.0.1", "port": 8000},
        "data": {"datasource": "toy"},
        "trainer": {"type": "basic", "rounds": 1},
        "algorithm": {"type": "fedavg"},
        "general": {"base_path": str(config_base)},
    }

    toml_writer.dump(config_data, config_path)

    monkeypatch.delenv("config_file", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            sys.argv[0],
            "--config",
            str(config_path),
            "--port",
            "9100",
            "--base",
            str(cli_base),
        ],
    )

    Config._instance = None
    if hasattr(Config, "args"):
        delattr(Config, "args")
    Config._cli_overrides = {}

    config = Config()

    assert config.server.port == 9100
    assert Config.server.port == 9100
    assert Config.args.port == 9100
    assert Config.params["base_path"] == str(cli_base)
    assert cli_base.is_dir()

    Config._instance = None
    if hasattr(Config, "args"):
        delattr(Config, "args")
    Config._cli_overrides = {}


def test_config_base_path_used_without_cli_override(tmp_path: Path, monkeypatch):
    config_base = tmp_path / "config_base"
    config_path = tmp_path / "config.toml"

    config_data = {
        "clients": {"type": "simple", "total_clients": 1, "per_round": 1},
        "server": {"address": "127.0.0.1", "port": 8000},
        "data": {"datasource": "toy"},
        "trainer": {"type": "basic", "rounds": 1},
        "algorithm": {"type": "fedavg"},
        "general": {"base_path": str(config_base)},
    }

    toml_writer.dump(config_data, config_path)

    monkeypatch.delenv("config_file", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            sys.argv[0],
            "--config",
            str(config_path),
        ],
    )

    Config._instance = None
    if hasattr(Config, "args"):
        delattr(Config, "args")
    Config._cli_overrides = {}

    config = Config()

    assert config.server.port == 8000
    assert Config.args.port is None
    assert Config.params["base_path"] == str(config_base)
    assert config_base.is_dir()

    Config._instance = None
    if hasattr(Config, "args"):
        delattr(Config, "args")
    Config._cli_overrides = {}
