"""Shared checkpoint names and explicit paths under a configured root.

Ordinary ASCII names retain their historical spelling. Names containing path
separators or other special characters use a reserved '~' plus SHA-256 digest.
The reserved prefix is itself encoded, so 'org/model', 'org_model' and a literal
encoded-looking name remain distinct. Call this on logical components, never
on an already assembled filename.
"""

import hashlib
import json
import re
from os import PathLike
from pathlib import Path


def checkpoint_component(value: str | int) -> str:
    """Encode a logical name as one collision-resistant filename component."""
    name = str(value)
    if name not in {"", ".", ".."} and re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        return name
    return "~" + hashlib.sha256(name.encode("utf-8")).hexdigest()


def checkpoint_name(*components: str | int, suffix: str = "") -> str:
    """Join logical components with historical underscores and an extension."""
    if not components or not re.fullmatch(r"(?:\.[A-Za-z0-9]+)*", suffix):
        raise ValueError("Checkpoint names require components and a file suffix")
    name = "_".join(checkpoint_component(item) for item in components) + suffix
    if len(name.encode("utf-8")) > 255:
        # Preserve long logical names without exceeding ordinary filesystem
        # component limits. Include boundaries, so distinct component tuples
        # cannot collide just because they share an underscore spelling.
        logical = json.dumps([str(item) for item in components], ensure_ascii=False)
        name = "~t" + hashlib.sha256(logical.encode("utf-8")).hexdigest() + suffix
        if len(name.encode("utf-8")) > 255:
            raise ValueError("Checkpoint suffix exceeds the filename component limit")
    return name


def checkpoint_path(root: str | PathLike, filename: str | PathLike) -> str:
    """Resolve an explicit relative filename within its root.

    Relative subdirectories are supported. Absolute filenames, traversal and
    symlinks that escape the selected root are rejected. Callers create parents
    only when writing; this function does not change the filesystem.
    """
    base = Path(root).resolve()
    relative = Path(filename)
    if relative.is_absolute():
        raise ValueError("Checkpoint filename must be relative to its root")
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or path == base:
        raise ValueError("Checkpoint filename must stay within its root")
    return str(path)
