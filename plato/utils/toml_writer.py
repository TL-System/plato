"""Minimal TOML writer to support configuration migrations and tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


def dumps(
    data: Mapping[str, Any],
    *,
    comments: Mapping[tuple[str, ...], list[str]] | None = None,
) -> str:
    """Serialize a mapping into TOML."""
    normalized = _normalize(data)
    comment_map = {tuple(path): lines[:] for path, lines in (comments or {}).items()}
    lines: list[str] = []
    if () in comment_map:
        _emit_comment(lines, comment_map.pop(()))
    _write_table(lines, normalized, [], comment_map)
    return "\n".join(lines) + "\n"


def dump(
    data: Mapping[str, Any],
    path: Path,
    *,
    comments: Mapping[tuple[str, ...], list[str]] | None = None,
) -> None:
    """Write TOML to disk."""
    path = Path(path)
    path.write_text(dumps(data, comments=comments), encoding="utf-8")


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        normalized = [_normalize(item) for item in value]
        types = {type(item) for item in normalized}
        if len(types) > 1:
            adjusted = []
            for item in normalized:
                if isinstance(item, Mapping) and item == {"null": True}:
                    adjusted.append(item)
                else:
                    adjusted.append({"value": item})
            return adjusted
        return normalized
    if value is None:
        return {"null": True}
    return value


def _write_table(
    lines: list[str],
    table: MutableMapping[str, Any],
    path: list[str],
    comment_map: Mapping[tuple[str, ...], list[str]],
) -> None:
    inline_items: list[tuple[str, Any]] = []
    tables: list[tuple[str, MutableMapping[str, Any]]] = []
    array_tables: list[tuple[str, Sequence[Any]]] = []

    for key, value in table.items():
        if isinstance(value, MutableMapping):
            if value == {"null": True}:
                inline_items.append((key, value))
            else:
                tables.append((key, value))
        elif (
            isinstance(value, list)
            and value
            and all(isinstance(item, MutableMapping) for item in value)
        ):
            array_tables.append((key, value))
        else:
            inline_items.append((key, value))

    for key, value in inline_items:
        _maybe_emit_comment(lines, comment_map, tuple(path + [key]))
        key_repr = _format_key(key)
        lines.append(f"{key_repr} = {_format_value(value)}")

    for key, value in tables:
        if lines and lines[-1] != "":
            lines.append("")
        header = _join(path + [key])
        _maybe_emit_comment(lines, comment_map, tuple(path + [key]))
        lines.append(f"[{header}]")
        _write_table(lines, value, path + [key], comment_map)

    for key, items in array_tables:
        if lines and lines[-1] != "":
            lines.append("")
        header = _join(path + [key])
        for index, item in enumerate(items):
            item_path = tuple(path + [key, str(index)])
            _maybe_emit_comment(lines, comment_map, item_path)
            lines.append(f"[[{header}]]")
            if not isinstance(item, MutableMapping):
                raise TypeError("Arrays of tables must contain mappings only.")
            _write_table(lines, item, path + [key, str(index)], comment_map)
            if index != len(items) - 1:
                lines.append("")


def _join(segments: Iterable[str]) -> str:
    parts = []
    for segment in segments:
        parts.append(_format_key(segment))
    return ".".join(parts)


def _needs_quotes(key: str) -> bool:
    return not key.replace("_", "").replace("-", "").isalnum()


def _format_key(key: str) -> str:
    return json.dumps(key) if _needs_quotes(key) else key


def _format_value(value: Any) -> str:
    if isinstance(value, dict):
        if value == {"null": True}:
            return "{ null = true }"
        items = ", ".join(f"{k} = {_format_value(v)}" for k, v in value.items())
        return f"{{ {items} }}"
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f"Unsupported value type for TOML serialization: {type(value)!r}")


def _emit_comment(lines: list[str], comments: list[str]) -> None:
    if not comments:
        return
    if lines and lines[-1] != "":
        lines.append("")
    elif not lines:
        lines.append("")
    for comment in comments:
        if comment:
            lines.append(f"# {comment}")
        else:
            lines.append("#")


def _maybe_emit_comment(
    lines: list[str],
    comment_map: Mapping[tuple[str, ...], list[str]],
    path: tuple[str, ...],
) -> None:
    if path in comment_map:
        _emit_comment(lines, comment_map[path])
