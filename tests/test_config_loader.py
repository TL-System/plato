"""Tests for the TOML configuration loader and ConfigNode helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
    assert Config._cli_overrides["port"] is True
    assert Config.args.port is None
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


def test_config_loads_evaluation_section(tmp_path: Path, monkeypatch):
    """Test that [evaluation] configuration is properly loaded."""
    config_base = tmp_path / "runtime"
    config_path = tmp_path / "config.toml"

    config_data = {
        "clients": {"type": "simple", "total_clients": 2, "per_round": 1},
        "server": {"address": "127.0.0.1", "port": 8000},
        "data": {"datasource": "MNIST"},
        "trainer": {"type": "basic", "rounds": 1, "epochs": 1, "batch_size": 10},
        "algorithm": {"type": "fedavg"},
        "evaluation": {
            "type": "nanochat_core",
            "max_per_task": 128,
            "bundle_dir": "/custom/path/to/nanochat",
        },
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
            "--base",
            str(config_base),
        ],
    )

    Config._instance = None
    if hasattr(Config, "args"):
        delattr(Config, "args")
    Config._cli_overrides = {}

    config = Config()

    assert hasattr(config, "evaluation")
    assert config.evaluation.type == "nanochat_core"
    assert config.evaluation.max_per_task == 128
    assert config.evaluation.bundle_dir == "/custom/path/to/nanochat"

    Config._instance = None
    if hasattr(Config, "args"):
        delattr(Config, "args")
    Config._cli_overrides = {}




def test_config_loads_smolvla_lerobot_parameter_contract(tmp_path: Path, monkeypatch):
    """SmolVLA/LeRobot config keys should round-trip through Config()."""
    config_base = tmp_path / "runtime"
    config_path = tmp_path / "smolvla_lerobot.toml"

    config_data = {
        "clients": {"type": "simple", "total_clients": 2, "per_round": 1},
        "server": {"address": "127.0.0.1", "port": 8000},
        "data": {"datasource": "LeRobot"},
        "trainer": {
            "type": "lerobot",
            "rounds": 1,
            "epochs": 1,
            "batch_size": 2,
            "model_type": "smolvla",
            "model_name": "smolvla",
        },
        "algorithm": {"type": "fedavg"},
        "parameters": {
            "policy": {
                "type": "smolvla",
                "path": "lerobot/smolvla_base",
                "finetune_mode": "adapter",
                "precision": "bf16",
                "device": "cuda",
            },
            "dataset": {
                "repo_id": "lerobot/pusht_image",
                "delta_timestamps": {
                    "observation_image": [-0.2, -0.1, 0.0],
                },
            },
            "transforms": {
                "image_size": [224, 224],
                "normalize": True,
                "interpolation": "bilinear",
            },
        },
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
            "--base",
            str(config_base),
        ],
    )

    Config._instance = None
    if hasattr(Config, "args"):
        delattr(Config, "args")
    Config._cli_overrides = {}

    config = Config()

    assert config.data.datasource == "LeRobot"
    assert config.trainer.type == "lerobot"
    assert config.trainer.model_type == "smolvla"
    assert config.trainer.model_name == "smolvla"

    assert config.parameters.policy.type == "smolvla"
    assert config.parameters.policy.path == "lerobot/smolvla_base"
    assert config.parameters.policy.finetune_mode == "adapter"
    assert config.parameters.policy.precision == "bf16"
    assert config.parameters.policy.device == "cuda"

    assert config.parameters.dataset.repo_id == "lerobot/pusht_image"
    assert config.parameters.dataset.delta_timestamps.observation_image == [
        -0.2,
        -0.1,
        0.0,
    ]

    assert config.parameters.transforms.image_size == [224, 224]
    assert config.parameters.transforms.normalize is True
    assert config.parameters.transforms.interpolation == "bilinear"

    assert config.parameters.policy._asdict() == {
        "type": "smolvla",
        "path": "lerobot/smolvla_base",
        "finetune_mode": "adapter",
        "precision": "bf16",
        "device": "cuda",
    }
    assert config.parameters.dataset._asdict() == {
        "repo_id": "lerobot/pusht_image",
        "delta_timestamps": {
            "observation_image": [-0.2, -0.1, 0.0],
        },
    }
    assert config.parameters.transforms._asdict() == {
        "image_size": [224, 224],
        "normalize": True,
        "interpolation": "bilinear",
    }

    Config._instance = None
    if hasattr(Config, "args"):
        delattr(Config, "args")
    Config._cli_overrides = {}


def test_is_central_server_requires_cross_silo_true(tmp_path: Path, monkeypatch):
    """Central-server detection should respect `cross_silo = false`."""
    config_path = tmp_path / "config.toml"
    config_data = {
        "clients": {"type": "simple", "total_clients": 1, "per_round": 1},
        "server": {"address": "127.0.0.1", "port": 8000},
        "data": {"datasource": "toy"},
        "trainer": {"type": "basic", "rounds": 1},
        "algorithm": {"type": "fedavg", "cross_silo": False},
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

    assert Config.is_central_server() is False
    assert getattr(config.algorithm, "cross_silo", False) is False

    Config._instance = None
    if hasattr(Config, "args"):
        delattr(Config, "args")
    Config._cli_overrides = {}


def test_toml_loader_allows_shared_includes(tmp_path: Path):
    """Shared include files should not be treated as circular includes."""
    common = tmp_path / "common.toml"
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    config_path = tmp_path / "config.toml"

    common.write_text("seed = 7\n", encoding="utf-8")
    first.write_text(
        """
include = "common.toml"
alpha = 1
""",
        encoding="utf-8",
    )
    second.write_text(
        """
include = "common.toml"
beta = 2
""",
        encoding="utf-8",
    )
    config_path.write_text(
        """
[runner]
include = ["first.toml", "second.toml"]
""",
        encoding="utf-8",
    )

    loader = TomlConfigLoader(config_path)
    config = loader.load()

    assert config["runner"]["seed"] == 7
    assert config["runner"]["alpha"] == 1
    assert config["runner"]["beta"] == 2


def test_toml_loader_detects_circular_includes(tmp_path: Path):
    """Mutually recursive includes should still raise a clear error."""
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"

    first.write_text('include = "second.toml"\n', encoding="utf-8")
    second.write_text('include = "first.toml"\n', encoding="utf-8")

    loader = TomlConfigLoader(first)
    with pytest.raises(ValueError, match="Circular include detected"):
        _ = loader.load()


def test_config_reload_discards_optional_sections_in_both_orders(tmp_path, monkeypatch):
    """Exercise successive production loads within one fixture lifetime (F13)."""
    required = {
        "clients": {"total_clients": 2, "per_round": 1},
        "server": {"address": "127.0.0.1", "port": 8000},
        "data": {"datasource": "MNIST"},
        "trainer": {"rounds": 1},
        "algorithm": {"type": "fedavg"},
    }
    optional = {
        "general": {"base_path": str(tmp_path / "from_config")},
        "evaluation": {"type": "nanochat_core"},
        "results": {"types": "round, loss", "result_path": "custom_results"},
        "parameters": {"optimizer": {"lr": 0.02}},
    }
    paths = [tmp_path / "minimal.toml", tmp_path / "optional.toml"]
    toml_writer.dump(required, paths[0])
    toml_writer.dump({**required, **optional}, paths[1])
    monkeypatch.delenv("config_file", raising=False)

    for reset_mode in ("legacy", "public"):
        for has_optional in (False, True, False, True, True, False):
            monkeypatch.setattr(
                sys, "argv", ["plato", "-c", str(paths[int(has_optional)]),
                              "--base", str(tmp_path / "cli")]
            )
            # No fixture runs between these transitions. Cover both the public
            # reset and historical callers which invalidate _instance directly.
            if reset_mode == "legacy":
                Config._instance = None
            else:
                Config.reset()
                assert all(not hasattr(Config, key) for key in optional)
                assert Config._cli_overrides == {}
                assert Config.client_sleep_times is None
            config = Config()
            assert all(hasattr(config, key) == has_optional for key in optional)
            assert Config.params["base_path"] == str(tmp_path / "cli")
            expected = (
                "round, loss" if has_optional else "round, accuracy, elapsed_time"
            )
            assert Config.params["result_types"] == expected
            assert Config.args.port is None


def test_speed_simulation_preserves_rng_and_seeded_values(temp_config):
    import numpy as np

    Config.clients.random_seed = 7
    Config.clients.max_sleep_time = 60
    Config.clients.simulation_distribution = Config.node_from_dict(
        {"distribution": "uniform", "low": 2.0, "high": 3.0}
    )
    before = np.random.get_state()
    actual = Config.simulate_client_speed()
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    # Independent fixed reference for NumPy's historical seed-7 draws.
    np.testing.assert_allclose(actual, [2.076308289373957, 2.7799187922401147])
    np.testing.assert_array_equal(Config.simulate_client_speed(), actual)


def test_nested_include_order_deep_overrides_and_list_concatenation(tmp_path):
    included = tmp_path / "included"
    included.mkdir()
    (included / "base.toml").write_text(
        'items = [1]\n[nested]\nkeep = 3\nreplace = 4\n'
    )
    (included / "second.toml").write_text(
        'include = "base.toml"\nitems = [2]\n[nested]\nreplace = 5\n'
    )
    root = tmp_path / "config.toml"
    root.write_text(
        'include = "included/second.toml"\nitems = [3]\n[nested]\nreplace = 6\n'
    )
    assert TomlConfigLoader(root).load() == {
        "items": [1, 2, 3], "nested": {"keep": 3, "replace": 6}
    }


@pytest.mark.parametrize("directive", ['include = 3', 'include = "missing.toml"'])
def test_invalid_include_fails_usefully(tmp_path, directive):
    root = tmp_path / "config.toml"
    root.write_text(directive)
    if directive.endswith("3"):
        with pytest.raises(TypeError, match="string or list"):
            TomlConfigLoader(root).load()
    else:
        with pytest.raises(FileNotFoundError, match="missing.toml"):
            TomlConfigLoader(root).load()


def test_environment_config_and_client_cli_overrides(temp_config, monkeypatch):
    # config_file is the established worker-process config channel, even when
    # --config also appears. Client --id keeps the edge-server --port visible.
    monkeypatch.setenv("config_file", str(Config.config_path))
    monkeypatch.setattr(sys, "argv", ["plato", "-c", "missing.toml", "-i", "2",
                                     "-p", "9001", "--server", "127.0.0.2:9002",
                                     "--base", Config.params["base_path"]])
    Config.reset()
    config = Config()
    assert Config.args.id == 2
    assert Config.args.port == 9001
    assert config.server.port == 9002
    assert config.server.address == "127.0.0.2"
