#!/usr/bin/env python
"""Convert YAML configuration files to TOML while preserving comments.

Run with `uv run --workspace tools python tools/convert_yaml_to_toml.py`.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from plato.utils import toml_writer


class IncludeRef(str):
    """Sentinel object to preserve `!include` references during conversion."""


yaml = YAML()
yaml.preserve_quotes = True


def include_constructor(loader, node):
    value = loader.construct_scalar(node)
    return IncludeRef(value)


yaml.constructor.add_constructor("!include", include_constructor)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert YAML files to TOML while carrying over comments."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=["configs", "examples", "tests"],
        help="Directories (relative to repo root) to scan for configuration files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files and report but do not write TOML output.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print files as they are processed."
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    processed = 0
    skipped = 0

    for target in args.targets:
        target_path = repo_root / target
        if not target_path.exists():
            continue
        for toml_path in target_path.rglob("*.toml"):
            yaml_rel = toml_path.relative_to(repo_root).with_suffix(".yml")
            yaml_text = _read_yaml_text(yaml_rel, repo_root)
            if yaml_text is None:
                skipped += 1
                continue

            data = yaml.load(yaml_text)
            comments = _collect_comments(data)
            plain_data = _to_plain_data(data)

            if args.verbose:
                print(f"Converting {yaml_rel} -> {toml_path.relative_to(repo_root)}")

            if not args.dry_run:
                toml_writer.dump(
                    plain_data,
                    toml_path,
                    comments=comments,
                )

            processed += 1

    if args.verbose:
        print(f"Processed {processed} files; skipped {skipped} (source missing).")


def _read_yaml_text(path: Path, repo_root: Path) -> str | None:
    """Return the YAML contents either from disk or the current git HEAD."""
    absolute = repo_root / path
    if absolute.exists():
        return absolute.read_text(encoding="utf-8")

    relative = path.as_posix()
    try:
    completed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None

    return completed.stdout


def _to_plain_data(node: Any) -> Any:
    """Recursively convert ruamel nodes (plus `!include`) into plain Python objects."""
    if isinstance(node, IncludeRef):
        value = str(node)
        if value.lower().endswith((".yml", ".yaml")):
            value = str(Path(value).with_suffix(".toml"))
        return {"include": value}

    if isinstance(node, CommentedMap):
        return {str(key): _to_plain_data(value) for key, value in node.items()}

    if isinstance(node, CommentedSeq):
        return [_to_plain_data(item) for item in node]

    return node


def _collect_comments(node: Any, path: Tuple[str, ...] = ()) -> Dict[tuple[str, ...], list[str]]:
    """Collect comment lines attached to each node keyed by its traversal path."""
    comments: Dict[tuple[str, ...], list[str]] = {}

    comment_attr = getattr(node, "ca", None)
    if (
        path == ()
        and comment_attr is not None
        and getattr(comment_attr, "comment", None)
    ):
        comment_lines, _ = _comment_info_from_parts(comment_attr.comment)
        if comment_lines:
            comments[path] = comment_lines

    if isinstance(node, CommentedMap):
        pending_follow: list[str] = []
        for key, value in node.items():
            key_path = path + (str(key),)
            entry = comment_attr.items.get(key) if comment_attr else None
            key_parts = _select_comment_parts(entry, (0, 1, 3))
            next_parts = _select_comment_parts(entry, (2,))

            key_lines, min_col = _comment_info_from_parts(key_parts)
            follow_lines, _ = _comment_info_from_parts(next_parts)

            lines_to_apply: list[str] = []
            if pending_follow:
                lines_to_apply.extend(pending_follow)
            if key_lines:
                lines_to_apply.extend(key_lines)

            if lines_to_apply:
                if (
                    min_col is not None
                    and min_col > 0
                    and isinstance(value, CommentedMap)
                    and value
                ):
                    first_key = next(iter(value.keys()), None)
                    if first_key is not None:
                        target_path = key_path + (str(first_key),)
                        existing = comments.get(target_path, [])
                        comments[target_path] = lines_to_apply + existing
                    else:
                        existing = comments.get(key_path, [])
                        comments[key_path] = (
                            existing + lines_to_apply if existing else lines_to_apply
                        )
                else:
                    existing = comments.get(key_path, [])
                    comments[key_path] = (
                        existing + lines_to_apply if existing else lines_to_apply
                    )

            pending_follow = follow_lines

            child_comments = _collect_comments(value, key_path)
            for child_path, child_lines in child_comments.items():
                if child_path in comments:
                    comments[child_path].extend(child_lines)
                else:
                    comments[child_path] = child_lines

    elif isinstance(node, CommentedSeq):
        for index, item in enumerate(node):
            item_path = path + (str(index),)
            entry = comment_attr.items.get(index) if comment_attr else None
            lines, _ = _comment_info_from_parts(entry)
            if lines:
                comments[item_path] = lines
            child_comments = _collect_comments(item, item_path)
            for child_path, child_lines in child_comments.items():
                if child_path in comments:
                    comments[child_path].extend(child_lines)
                else:
                    comments[child_path] = child_lines

    return comments


def _extract_comment_parts(comment_accessor: Any) -> Iterable[Any]:
    """Extract components from a ruamel comment accessor."""
    if not comment_accessor:
        return []
    if isinstance(comment_accessor, list):
        return comment_accessor
    return [comment_accessor]


def _select_comment_parts(entry: Any, indices: tuple[int, ...]) -> list[Any]:
    if not entry:
        return []
    parts: list[Any] = []
    for index in indices:
        if index < len(entry) and entry[index] is not None:
            parts.append(entry[index])
    return parts


def _comment_info_from_parts(parts: Any) -> tuple[list[str], int | None]:
    """Normalise comment tokens into (lines, minimum column) tuples."""
    lines: list[str] = []
    min_col: int | None = None

    def _consume(part: Any) -> None:
        nonlocal min_col
        if part is None:
            return
        if isinstance(part, (list, tuple)):
            for item in part:
                _consume(item)
            return
        value = getattr(part, "value", None)
        if value is None and isinstance(part, str):
            value = part
        if value is None:
            return
        column = getattr(part, "column", None)
        if column is not None:
            min_col = column if min_col is None else min(min_col, column)
        for raw_line in value.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("#"):
                lines.append(stripped[1:].strip())

    _consume(parts)
    return lines, min_col


if __name__ == "__main__":
    main()
