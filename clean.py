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
    """Yield directories whose names match the configured target set."""
    root = root.resolve()
    runtime_root = (root / "runtime").resolve()
    walk_roots = []

    if runtime_root.exists():
        walk_roots.append(runtime_root)
    if root.name == "runtime" and root.is_dir():
        walk_roots.append(root)

    if not walk_roots:
        return

    seen: set[Path] = set()
    for walk_root in walk_roots:
        resolved_root = walk_root.resolve()
        if resolved_root in seen:
            continue
        seen.add(resolved_root)
        for current, dirnames, _ in os.walk(resolved_root):
            dirnames[:] = [dirname for dirname in dirnames if dirname not in EXCLUDED_NAMES]
            for dirname in dirnames:
                if dirname in TARGET_NAMES:
                    yield Path(current) / dirname


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
