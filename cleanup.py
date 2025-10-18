#!/usr/bin/env python3
"""Clean temporary directories across the Plato codebase."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable

RUNTIME_NAME = "runtime"
PYCACHE_NAME = "__pycache__"
EXCLUDED_NAMES = {".venv"}


def find_runtime_roots(root: Path) -> list[Path]:
    """Return all ``runtime`` directories located beneath ``root``."""
    root = root.resolve()

    runtime_roots: list[Path] = []
    seen_runtime: set[Path] = set()

    def add_runtime(candidate: Path) -> None:
        try:
            resolved = candidate.resolve()
        except OSError:
            return
        if resolved in seen_runtime or not candidate.is_dir():
            return
        seen_runtime.add(resolved)
        runtime_roots.append(candidate)

    if root.name == RUNTIME_NAME:
        add_runtime(root)

    for current, dirnames, _ in os.walk(root, topdown=True):
        pruned_dirs = []
        for dirname in dirnames:
            if dirname in EXCLUDED_NAMES:
                continue
            if dirname == RUNTIME_NAME:
                add_runtime(Path(current) / dirname)
                continue
            if dirname == PYCACHE_NAME:
                continue
            pruned_dirs.append(dirname)
        dirnames[:] = pruned_dirs

    return runtime_roots


def iter_pycache_directories(root: Path) -> Iterable[Path]:
    """Yield all ``__pycache__`` directories under ``root`` (excluding excluded paths)."""
    root = root.resolve()

    if root.name == PYCACHE_NAME and root.is_dir():
        yield root

    seen_pycache: set[Path] = set()
    for current, dirnames, _ in os.walk(root, topdown=True):
        pruned_dirs = []
        for dirname in dirnames:
            if dirname in EXCLUDED_NAMES:
                continue
            if dirname == PYCACHE_NAME:
                candidate = Path(current) / dirname
                resolved_candidate = candidate.resolve()
                if resolved_candidate in seen_pycache:
                    continue
                seen_pycache.add(resolved_candidate)
                yield candidate
                continue
            pruned_dirs.append(dirname)
        dirnames[:] = pruned_dirs


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


def remove_directory(path: Path) -> bool:
    """Remove the directory at ``path`` entirely. Returns ``True`` on success."""
    try:
        shutil.rmtree(path)
        return True
    except OSError as exc:
        print(f"Failed to remove {path}: {exc}")
        return False


def resolve_root(path_str: str | None) -> Path:
    """Resolve the repository root to clean under."""
    if path_str is None:
        return Path(__file__).resolve().parent
    return Path(path_str).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove Plato runtime directories (and their contents) under the "
            "given root directory and delete any __pycache__ folders."
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

    runtime_roots = find_runtime_roots(root)
    runtime_total = len(runtime_roots)
    runtime_removed = 0
    fallback_dirs = 0
    fallback_items = 0

    for runtime_dir in runtime_roots:
        if remove_directory(runtime_dir):
            print(f"Deleted runtime directory {runtime_dir}")
            runtime_removed += 1
            continue

        cleared = clean_directory(runtime_dir)
        print(
            f"Failed to delete {runtime_dir}; cleared {cleared} items instead."
        )
        fallback_dirs += 1
        fallback_items += cleared

    if runtime_total == 0:
        print("No runtime directories found.")
    else:
        print(
            f"Removed {runtime_removed} of {runtime_total} runtime directories."
        )
        if fallback_dirs:
            print(
                f"Cleared {fallback_items} items in "
                f"{fallback_dirs} undeleted runtime directories."
            )

    if not root.exists():
        print("Root directory removed; skipped __pycache__ cleanup.")
        return

    pycache_removed = 0
    pycache_total = 0
    for pycache_dir in iter_pycache_directories(root):
        if remove_directory(pycache_dir):
            print(f"Deleted __pycache__ directory {pycache_dir}")
            pycache_removed += 1
        pycache_total += 1

    if pycache_total == 0:
        print("No __pycache__ directories found.")
    else:
        print(f"Removed {pycache_removed} of {pycache_total} __pycache__ directories.")


if __name__ == "__main__":
    main()
