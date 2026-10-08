"""Offline utility behavior, numeric stability, and filesystem persistence."""

import csv
import importlib
import tomllib

import numpy as np
import pytest
import torch

from plato.utils import csv_processor, data_loaders, fonts, toml_writer, unary_encoding


@pytest.mark.parametrize(
    "loader_cls", [data_loaders.ParallelDataLoader, data_loaders.SequentialDataLoader]
)
def test_empty_compound_loader_terminates(loader_cls):
    loader = loader_cls([None])
    assert len(loader) == 0
    assert list(loader) == []


def test_parameter_count_utility_import_never_downloads_models(monkeypatch):
    def network_access(*_args, **_kwargs):
        raise AssertionError("Import attempted a network model download")

    monkeypatch.setattr(torch.hub, "load", network_access)
    module = importlib.import_module("plato.utils.count_parameters")
    assert callable(module.count_parameters)


@pytest.mark.parametrize(
    "method",
    [unary_encoding.symmetric_unary_encoding, unary_encoding.optimized_unary_encoding],
)
def test_unary_encoding_large_epsilon_remains_finite(method):
    values = np.array([0, 1] * 1000)
    result = method(values, 2000)
    assert set(result.tolist()) <= {0, 1}
    assert np.all(result[values == 0] == 0)
    if method == unary_encoding.symmetric_unary_encoding:
        assert np.all(result[values == 1] == 1)
    else:
        assert abs(result[values == 1].mean() - 0.5) < 0.05


def test_csv_extension_preserves_rows_and_quotes(tmp_path):
    path = tmp_path / "results.csv"
    csv_processor.initialize_csv(str(path), ["round", "note"], str(tmp_path))
    csv_processor.write_csv(str(path), [1, "commas, and newlines\n"])
    csv_processor.expand_csv_columns(str(path), ["accuracy", "round"])
    csv_processor.write_csv(str(path), [2, "ok", 0.75])
    with path.open(newline="") as handle:
        assert list(csv.reader(handle)) == [
            ["round", "note", "accuracy"],
            ["1", "commas, and newlines\n", ""],
            ["2", "ok", "0.75"],
        ]


def test_toml_config_roundtrip_matches_independent_parser():
    config = {
        "server": {"address": "127.0.0.1", "port": 8000},
        "trainer": {"epochs": 2, "train": True, "rates": [0.1, 0.2]},
        "literal.dot": {"quote": 'line\n"quoted"'},
    }
    assert tomllib.loads(toml_writer.dumps(config)) == config


def test_logging_font_invalid_choice_is_explicit():
    assert fonts.colourize("test", "green").endswith("test\033[0m")
    with pytest.raises(ValueError, match="not supported"):
        fonts.colourize("test", "unknown")


def test_toml_nested_array_tables_and_literal_keys_keep_structure():
    config = {
        "café": "literal Unicode key",
        "items": [
            {"name": "first", "child": {"literal.dot": 1}},
            {"name": "second", "child": {"x": [1, 2]}},
        ],
        "mixed": [1, "two"],
    }
    # Mixed-type arrays retain the writer's established wrapper convention.
    expected = {**config, "mixed": [{"value": 1}, {"value": "two"}]}
    assert tomllib.loads(toml_writer.dumps(config)) == expected


@pytest.mark.parametrize(
    "config",
    [
        {"🚀": 1},
        {"title": "experiment 🚀"},
        {
            "tables": [
                {"𠮷": {"literal.dot": [1, 2], "title": "💡"}},
                {"𠮷": {"literal.dot": [3]}},
            ]
        },
        {"delete\x7f": "has\x7f and 🚀"},
        {"delete\x7f": "has\x7f"},
    ],
)
def test_toml_supplementary_unicode_and_control_escapes_roundtrip(config, tmp_path):
    comments: dict[tuple[str, ...], list[str]] = {(): ["UTF-8 🚀"]}
    wire = toml_writer.dumps(config, comments=comments)
    assert tomllib.loads(wire) == config
    path = tmp_path / "unicode.toml"
    toml_writer.dump(config, path, comments=comments)
    with path.open("rb") as handle:
        assert tomllib.load(handle) == config
