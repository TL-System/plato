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
    # Model writers append '.pkl' for history. Reserve that space when naming
    # the primary file, rather than discovering an overlong sidecar afterward.
    limit = 251 if suffix in {".safetensors", ".pth"} else 255
    if len(name.encode("utf-8")) > limit:
        # Preserve long logical names without exceeding ordinary filesystem
        # component limits. Include boundaries, so distinct component tuples
        # cannot collide just because they share an underscore spelling.
        logical = json.dumps([str(item) for item in components], ensure_ascii=False)
        name = "~t" + hashlib.sha256(logical.encode("utf-8")).hexdigest() + suffix
        if len(name.encode("utf-8")) > limit:
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


def snapshot_details(filename: str) -> tuple[int, int, float] | None:
    """Identify owned epoch/time snapshots, including historical cleanup sidecars.

    Safetensors readers still select their supported format explicitly. This
    recognizes historical Torch names for cleanup without broadening readers.
    """
    match = re.fullmatch(
        r"(?P<client>\d+)_(?P<epoch>\d+)_"
        r"(?P<time>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
        r"\.(?:safetensors|pth)(?:\.pkl)?", filename,
    )
    if match is None:
        return None
    return int(match["client"]), int(match["epoch"]), float(match["time"])


def checkpoint_sidecar(
    root: str | PathLike, primary: str | PathLike, suffix: str = ".pkl"
) -> str:
    """Attach a suffix to a resolved checkpoint, checking sidecar containment.

    A safe primary path does not imply its separately named sidecar is safe:
    an existing history symlink must also remain within the selected root.
    """
    if not re.fullmatch(r"(?:\.[A-Za-z0-9]+)+", suffix):
        raise ValueError("Checkpoint sidecars require a file suffix")
    base = Path(root).resolve()
    path = Path(primary)
    if not path.is_absolute():
        path = Path(checkpoint_path(base, path))
    return checkpoint_path(base, str(path.relative_to(base)) + suffix)
