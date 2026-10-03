"""Filesystem contracts that do not require the optional evaluation backend."""

from __future__ import annotations

import errno
import shutil
import stat
from pathlib import Path

import pytest

from plato.config import Config
from plato.evaluators.base import EvaluationInput
from plato.evaluators.lighteval import _materialize_model_reference
from tests.llm_eval.helpers import snapshot_directory


@pytest.mark.parametrize("tokenizer_source", ["same", "separate", "hub"])
def test_local_references_are_independent_and_source_preserving(
    temp_config, tmp_path, monkeypatch, tokenizer_source
):
    model_directory = tmp_path / "model"
    nested = model_directory / "nested"
    nested.mkdir(parents=True)
    (model_directory / "config.json").write_text("{}")
    (nested / "readonly.txt").write_text("preserved")
    (nested / "readonly.txt").chmod(0o444)
    nested.chmod(0o555)
    blob = tmp_path / "weights-blob"
    blob.write_bytes(b"independent artifact")
    (model_directory / "weights.bin").symlink_to(blob)
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir()
    (tokenizer_directory / "tokenizer.json").write_text("{}")
    sentinel = tmp_path / "unrelated"
    sentinel.write_text("keep")
    original_model = snapshot_directory(model_directory)
    original_tokenizer = snapshot_directory(tokenizer_directory)
    monkeypatch.chdir(tmp_path)
    Config().trainer.model_name = "model"
    Config().trainer.tokenizer_name = {
        "same": "./model",
        "separate": "tokenizer",
        "hub": "example/tokenizer",
    }[tokenizer_source]
    copied_roots = []
    original_copytree = shutil.copytree

    def record_copy(source, destination, *args, **kwargs):
        if Path(source) in (model_directory, tokenizer_directory):
            copied_roots.append(Path(source))
        return original_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copytree", record_copy)
    with _materialize_model_reference(EvaluationInput(model=object())) as reference:
        copied_model = Path(reference.model_name)
        assert copied_model != model_directory
        assert copied_model.stat().st_mode & stat.S_IWUSR
        assert (copied_model / "nested" / "readonly.txt").read_text() == "preserved"
        assert not (copied_model / "weights.bin").is_symlink()
        assert not (copied_model / "weights.bin").samefile(blob)
        (copied_model / "weights.bin").write_bytes(b"own copy changed")
        assert blob.read_bytes() == b"independent artifact"
        if tokenizer_source == "same":
            assert reference.tokenizer_name == reference.model_name
        elif tokenizer_source == "separate":
            assert reference.tokenizer_name is not None
            assert Path(reference.tokenizer_name) != tokenizer_directory
            assert Path(reference.tokenizer_name, "tokenizer.json").read_text() == "{}"
        else:
            assert reference.tokenizer_name == "example/tokenizer"

    assert not copied_model.exists()
    assert copied_roots == (
        [model_directory, tokenizer_directory]
        if tokenizer_source == "separate"
        else [model_directory]
    )
    assert snapshot_directory(model_directory) == original_model
    assert snapshot_directory(tokenizer_directory) == original_tokenizer
    assert sentinel.read_text() == "keep"
    assert Config().trainer.model_name == "model"


def test_broken_source_link_propagates_and_cleans_partial_copy(
    temp_config, tmp_path, monkeypatch
):
    from plato.evaluators import lighteval

    source = tmp_path / "model"
    source.mkdir()
    (source / "valid.bin").write_bytes(b"valid")
    (source / "broken.bin").symlink_to(tmp_path / "missing")
    Config().trainer.model_name = str(source)
    Config().trainer.tokenizer_name = str(source)
    before = snapshot_directory(source)
    owned = []
    original_temporary_directory = lighteval.tempfile.TemporaryDirectory

    def record_owned_directory(*args, **kwargs):
        directory = original_temporary_directory(*args, **kwargs)
        owned.append(Path(directory.name))
        return directory

    monkeypatch.setattr(
        lighteval.tempfile, "TemporaryDirectory", record_owned_directory
    )
    with pytest.raises(shutil.Error, match="broken.bin"):
        with _materialize_model_reference(EvaluationInput(model=object())):
            pytest.fail("A failed copy must not fall back to the original source.")

    assert len(owned) == 1
    assert not owned[0].exists()
    assert snapshot_directory(source) == before


def test_hub_reference_remains_unchanged(temp_config):
    Config().trainer.model_name = "example/model"
    Config().trainer.tokenizer_name = "example/tokenizer"

    with _materialize_model_reference(EvaluationInput(model=object())) as reference:
        assert reference.model_name == "example/model"
        assert reference.tokenizer_name == "example/tokenizer"


@pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EACCES])
def test_destination_copy_error_keeps_original_cause_and_cleans_owned_paths(
    temp_config, tmp_path, monkeypatch, error_number
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"source preserved")
    Config().trainer.model_name = str(source)
    Config().trainer.tokenizer_name = str(source)
    before = snapshot_directory(source)
    sentinel = tmp_path / "unrelated"
    sentinel.write_text("keep")
    failure = OSError(error_number, "Injected destination copy failure")
    owned = []

    def fail_copy(source_directory, destination):
        destination = Path(destination)
        destination.mkdir()
        shutil.copy2(Path(source_directory, "weights.bin"), destination / "partial.bin")
        owned.append(destination)
        raise failure

    monkeypatch.setattr(shutil, "copytree", fail_copy)
    with pytest.raises(OSError) as error:
        with _materialize_model_reference(EvaluationInput(model=object())):
            pytest.fail("A failed copy must not evaluate the original source.")

    assert error.value is failure
    assert len(owned) == 1
    assert not owned[0].parent.exists()
    assert snapshot_directory(source) == before
    assert sentinel.read_text() == "keep"
