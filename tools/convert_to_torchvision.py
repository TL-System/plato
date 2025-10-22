#!/usr/bin/env python3
"""
Normalize configuration files so torchvision datasets are loaded via the generic
Torchvision datasource.

The script scans TOML files recursively (current directory by default) and
rewrites datasource declarations for well-known torchvision datasets so that
each block includes the required ``dataset_name`` and ``download`` fields as
well as dataset-specific options (for example, the balanced EMNIST split).

Usage examples
==============

Dry run showing which files would be updated::

    uv run --workspace tools python tools/convert_to_torchvision.py --dry-run

Apply the conversion to a particular subtree::

    uv run --workspace tools python tools/convert_to_torchvision.py configs/MNIST

The script is idempotent and can safely be re-run after manual edits.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset-specific defaults to inject next to the datasource line."""

    dataset_name: str
    extra_lines: Sequence[str] = ()
    include_download: bool = True


DATASET_SPECS: dict[str, DatasetSpec] = {
    "MNIST": DatasetSpec("MNIST"),
    "FashionMNIST": DatasetSpec("FashionMNIST"),
    "EMNIST": DatasetSpec("EMNIST", extra_lines=('dataset_kwargs = { split = "balanced" }',)),
    "CIFAR10": DatasetSpec("CIFAR10"),
    "CIFAR100": DatasetSpec("CIFAR100"),
    "STL10": DatasetSpec("STL10"),
    "CelebA": DatasetSpec(
        "CelebA",
        extra_lines=('dataset_kwargs = { target_type = ["attr", "identity"] }',),
    ),
}

DATASOURCE_LINE_RE = re.compile(
    r'^(?P<indent>\s*)(?P<key>datasource)\s*=\s*["\'](?P<dataset>[A-Za-z0-9_]+)["\']'
    r'(?P<comment>\s*(?:#.*)?)\s*$'
)


def iter_toml_files(root: Path) -> Iterable[Path]:
    """Yield every TOML file under ``root`` recursively."""
    yield from root.rglob("*.toml")


def _line_ending(line: str) -> str:
    """Return the newline characters used by ``line`` (if any)."""
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    if line.endswith("\r"):
        return "\r"
    return ""


def _build_insert_lines(spec: DatasetSpec, indent: str, newline: str) -> List[str]:
    """Construct the lines that must appear after the datasource entry."""
    terminator = newline or "\n"
    insert_lines = [f'{indent}dataset_name = "{spec.dataset_name}"{terminator}']
    if spec.include_download:
        insert_lines.append(f"{indent}download = true{terminator}")
    for extra in spec.extra_lines:
        insert_lines.append(f"{indent}{extra}{terminator}")
    return insert_lines


def transform_lines(lines: List[str]) -> tuple[List[str], bool]:
    """Apply datasource rewrites to the provided list of lines."""
    changed = False
    i = 0
    while i < len(lines):
        raw_line = lines[i]
        match = DATASOURCE_LINE_RE.match(raw_line.rstrip("\n\r"))
        if not match:
            i += 1
            continue

        dataset = match.group("dataset")
        spec = DATASET_SPECS.get(dataset)
        if spec is None:
            i += 1
            continue

        indent = match.group("indent") or ""
        comment = match.group("comment") or ""
        newline = _line_ending(raw_line)
        terminator = newline or "\n"

        # Update the datasource line itself.
        lines[i] = f'{indent}datasource = "Torchvision"{comment}{terminator}'

        # Remove any existing dataset-specific keys so we can reinsert in normal order.
        keys_to_remove = ["dataset_name"]
        if spec.include_download:
            keys_to_remove.append("download")
        for extra in spec.extra_lines:
            key = extra.split("=", 1)[0].strip()
            keys_to_remove.append(key)

        key_patterns = [
            re.compile(rf"^\s*{re.escape(key)}\s*=") for key in keys_to_remove
        ]

        j = i + 1
        section_end = j
        while j < len(lines):
            candidate = lines[j]
            stripped = candidate.strip()
            if stripped == "" or stripped.startswith("#"):
                j += 1
                continue
            if stripped.startswith("["):
                break
            if any(pattern.match(candidate) for pattern in key_patterns):
                del lines[j]
                changed = True
                continue
            j += 1
        section_end = j

        # Insert the canonical lines immediately after the datasource entry.
        insert_lines = _build_insert_lines(spec, indent, newline)
        lines[i + 1 : i + 1] = insert_lines
        changed = True
        # Skip past the newly inserted block.
        i += 1 + len(insert_lines)
        # Account for lines removed before the next section.
        if section_end < i:
            i = section_end
    return lines, changed


def convert_file(path: Path, *, dry_run: bool) -> bool:
    """Convert a single TOML file, returning True when the file changes."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    updated_lines, changed = transform_lines(lines)
    if changed and not dry_run:
        path.write_text("".join(updated_lines), encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite datasource blocks to use the generic Torchvision loader."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        type=Path,
        help="Root directory to scan (defaults to current working directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report files that would change without modifying them.",
    )
    args = parser.parse_args()

    root = args.path.resolve()
    if not root.exists():
        parser.error(f"Path does not exist: {root}")

    updated_files: list[Path] = []

    for toml_file in iter_toml_files(root):
        if convert_file(toml_file, dry_run=args.dry_run):
            updated_files.append(toml_file)

    if updated_files:
        header = "Would update" if args.dry_run else "Updated"
        print(f"{header} {len(updated_files)} file(s):", file=sys.stderr)
        for file_path in updated_files:
            print(f"  {file_path}", file=sys.stderr)
    else:
        print("No datasource entries required conversion.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
