"""Cleanup must not traverse directory aliases outside the chosen root."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "plato_cleanup", Path(__file__).parents[2] / "cleanup.py"
)
assert _SPEC is not None and _SPEC.loader is not None
cleanup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cleanup)


def test_symlink_runtime_fallback_never_clears_external_files(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    external = tmp_path / "external"
    root.mkdir()
    external.mkdir()
    protected = external / "checkpoint"
    protected.write_text("preserve")
    (root / "runtime").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(
        cleanup, "parse_args", lambda: type("Args", (), {"root": str(root)})
    )
    cleanup.main()
    assert protected.read_text() == "preserve"


def test_cleanup_removes_only_runtime_and_pycache(tmp_path):
    runtime = tmp_path / "example" / "runtime"
    pycache = tmp_path / "example" / "__pycache__"
    preserved = tmp_path / ".venv" / "runtime"
    for directory in (runtime, pycache, preserved):
        directory.mkdir(parents=True)
        (directory / "file").write_text("data")
    (tmp_path / "unrelated").write_text("preserve")
    assert cleanup.find_runtime_roots(tmp_path) == [runtime]
    assert cleanup.remove_directory(runtime)
    for directory in cleanup.iter_pycache_directories(tmp_path):
        assert cleanup.remove_directory(directory)
    assert not runtime.exists()
    assert not pycache.exists()
    assert (preserved / "file").read_text() == "data"
    assert (tmp_path / "unrelated").read_text() == "preserve"
