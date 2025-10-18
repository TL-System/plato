#!/usr/bin/env python3
"""Clean temporary directories across the Plato codebase."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable


TARGET_NAMES = {"results", "data", "checkpoints", "mpc_data"}
EXCLUDED_NAMES = {".venv"}


def iter_target_directories(root: Path) -> Iterable[Path]:
    """Yield target directories that live under any runtime folders within ``root``."""
    root = root.resolve()

    runtime_roots: list[Path] = []
    seen_runtime: set[Path] = set()

    def add_runtime(candidate: Path) -> None:
        resolved = candidate.resolve()
        if resolved in seen_runtime or not candidate.is_dir():
            return
        seen_runtime.add(resolved)
        runtime_roots.append(candidate)

    if root.name == "runtime":
        add_runtime(root)

    for current, dirnames, _ in os.walk(root, topdown=True):
        pruned_dirs = []
        for dirname in dirnames:
            if dirname in EXCLUDED_NAMES:
                continue
            if dirname == "runtime":
                add_runtime(Path(current) / dirname)
                continue
            pruned_dirs.append(dirname)
        dirnames[:] = pruned_dirs

    seen_targets: set[Path] = set()
    for walk_root in runtime_roots:
        for current, dirnames, _ in os.walk(walk_root):
            dirnames[:] = [dirname for dirname in dirnames if dirname not in EXCLUDED_NAMES]
            for dirname in dirnames:
                if dirname in TARGET_NAMES:
                    candidate = Path(current) / dirname
                    resolved_candidate = candidate.resolve()
                    if resolved_candidate in seen_targets:
                        continue
                    seen_targets.add(resolved_candidate)
                    yield candidate


def clean_directory(path: Path) -> int:
    """Remove all contents of the directory at ``path``. Returns items deleted."""
    removed = 0
    for child in path.iterdir():
        try:
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                continue
            removed += 1
        except OSError as exc:
            print(f"Failed to remove {child}: {exc}")
    return removed


def resolve_root(path_str: str | None) -> Path:
    """Resolve the repository root to clean under."""
    if path_str is None:
        return Path(__file__).resolve().parent
    return Path(path_str).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove contents of temporary directories (results, models, "
            "checkpoints, mpc_data) under the given root Plato directory."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        help="Optional root directory to scan (defaults to script location).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_root(args.root)

    if not root.is_dir():
        raise SystemExit(f"Root path is not a directory: {root}")

    print(f"Cleaning temporary directories under: {root}")

    total_removed = 0
    total_dirs = 0
    for target in iter_target_directories(root):
        removed = clean_directory(target)
        print(f"Cleared {removed} items in {target}")
        total_removed += removed
        total_dirs += 1

    if total_dirs == 0:
        print("No target directories found.")
    else:
        print(f"Finished cleaning {total_dirs} directories; removed {total_removed} items.")


if __name__ == "__main__":
    main()
