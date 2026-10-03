"""Contained extraction for dataset and evaluation TAR/ZIP bundles.

The destination must not be modified concurrently by another writer. All member
paths are checked before extraction, including existing destination symlinks.
"""

from __future__ import annotations

import os
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath


class UnsafeArchiveError(ValueError):
    """An archive member would violate the extraction boundary."""


def _member_path(name: str, root: Path) -> Path:
    path = PurePosixPath(name)
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or path.is_absolute()
        or PureWindowsPath(name).drive
        or ".." in path.parts
    ):
        raise UnsafeArchiveError(
            f"Unsafe archive path (absolute or traversal): {name!r}"
        )
    target = root.joinpath(*path.parts)
    _check_existing_links(target, root)
    return target


def _check_existing_links(target: Path, root: Path) -> None:
    current = root
    for part in target.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise UnsafeArchiveError(f"Unsafe existing destination symlink: {current}")


def extract_archive(
    source: str | os.PathLike[str], destination: str | os.PathLike[str]
) -> None:
    """Extract a TAR/ZIP after preflight; reject unsafe names, links and devices.

    Safe TAR links may point to regular files inside the destination. Members
    through archive symlinks and link chains are rejected. ZIP symlink entries
    are rejected rather than silently materialized as files. Rejection does not
    promise rollback of filesystem or corrupt-data errors during extraction.
    """
    root = Path(destination)
    if root.is_symlink():
        raise UnsafeArchiveError(f"Unsafe destination symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            for member in members:
                _member_path(member.filename, root)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise UnsafeArchiveError(
                        f"Unsafe ZIP symlink entry: {member.filename!r}"
                    )
            archive.extractall(root)
        return

    with tarfile.open(source, "r:*") as archive:
        members = archive.getmembers()
        paths = {member.name: _member_path(member.name, root) for member in members}
        symlinks = {paths[member.name] for member in members if member.issym()}
        for member in members:
            target = paths[member.name]
            if any(
                link == target and not member.issym() or link in target.parents
                for link in symlinks
            ):
                raise UnsafeArchiveError(
                    f"Unsafe member through archive symlink: {member.name!r}"
                )
            if member.issym() or member.islnk():
                link = member.linkname
                if (
                    not link
                    or "\x00" in link
                    or "\\" in link
                    or PurePosixPath(link).is_absolute()
                    or PureWindowsPath(link).drive
                ):
                    raise UnsafeArchiveError(f"Unsafe absolute archive link: {link!r}")
                link_base = target.parent if member.issym() else root
                linked = (link_base / link).resolve()
                if not linked.is_relative_to(root):
                    raise UnsafeArchiveError(
                        f"Unsafe archive link outside destination: {link!r}"
                    )
                _check_existing_links(link_base / link, root)
                if any(
                    other == linked or other in linked.parents for other in symlinks
                ):
                    raise UnsafeArchiveError(f"Unsafe archive link chain: {link!r}")
            elif not (member.isfile() or member.isdir()):
                raise UnsafeArchiveError(f"Unsafe special TAR member: {member.name!r}")
            # Explicit on both supported Python versions; use the standard data
            # filter for ownership/mode handling as well as a second path check.
            tarfile.data_filter(member, str(root))
        archive.extractall(root, members=members, filter="data")
