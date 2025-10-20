"""Tests for the TOML configuration loader and ConfigNode helpers."""

from __future__ import annotations

from pathlib import Path

from plato.config import Config, ConfigNode, TomlConfigLoader


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
