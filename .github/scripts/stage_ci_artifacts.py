"""Stage CI evidence without following pytest aliases or copying unit fixtures."""

import argparse
import errno
import hashlib
import json
import os
import stat
from pathlib import Path

MANIFEST = "artifact-selection.json"
UNIT_PARTITIONS = {"base-tmp", "mandatory-tmp"}
DIAGNOSTIC_SUFFIXES = {
    ".csv",
    ".json",
    ".log",
    ".stderr",
    ".stdout",
    ".toml",
    ".txt",
    ".xml",
}
DIAGNOSTIC_LIMIT = 1024 * 1024
EXPECTED_RECORDS = (
    "commit.txt",
    "submodules.txt",
    "lock.txt",
    "python.txt",
    "uv.txt",
    "base-packages.txt",
    "mandatory-packages.txt",
    "base-collection.log",
    "base.log",
    "base.xml",
    "missing-extra.log",
    "mandatory.log",
    "mandatory.xml",
    "coverage.json",
    "runtime.log",
    "runtime.xml",
    "ruff.log",
    "ty.log",
)


def stage_artifacts(source: Path, destination: Path) -> dict[str, object]:
    """Copy regular evidence files and report exclusions and interrupted reads."""
    source = source.absolute()
    destination = destination.absolute()
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("The staging destination must be outside the source.")
    destination.mkdir(parents=True, exist_ok=False)
    included = []
    omitted = []
    issues = []

    def omit(relative: Path, reason: str, size: int | None = None) -> None:
        record = {"path": relative.as_posix(), "reason": reason}
        if size is not None:
            record["bytes"] = size
        omitted.append(record)

    def read_issue(relative: Path, operation: str, error: OSError) -> None:
        issues.append(
            {
                "path": relative.as_posix(),
                "operation": operation,
                "errno": error.errno,
                "error": error.strerror,
            }
        )

    def copy_record(directory_fd: int, relative: Path) -> None:
        target = destination / relative
        try:
            descriptor = os.open(
                relative.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
        except OSError as error:
            omit(relative, "file_unavailable_or_replaced_before_copy")
            read_issue(relative, "open_file_without_following_links", error)
            return
        try:
            with os.fdopen(descriptor, "rb") as incoming:
                before = os.fstat(incoming.fileno())
                if not stat.S_ISREG(before.st_mode):
                    omit(relative, "not_a_regular_file")
                    return
                if (
                    relative.parts[0] in UNIT_PARTITIONS
                    and before.st_size > DIAGNOSTIC_LIMIT
                ):
                    omit(relative, "unit_diagnostic_exceeds_size_limit", before.st_size)
                    return
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                copied = 0
                remaining = before.st_size
                with target.open("xb") as outgoing:
                    while remaining:
                        chunk = incoming.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        outgoing.write(chunk)
                        digest.update(chunk)
                        copied += len(chunk)
                        remaining -= len(chunk)
                after = os.fstat(incoming.fileno())
                included.append(
                    {
                        "path": relative.as_posix(),
                        "bytes": copied,
                        "sha256": digest.hexdigest(),
                    }
                )
                if copied != before.st_size or (before.st_size, before.st_mtime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    issues.append(
                        {
                            "path": relative.as_posix(),
                            "operation": "copy_snapshot",
                            "error": "Source changed while copying; retained snapshot.",
                        }
                    )
        except OSError as error:
            target.unlink(missing_ok=True)
            omit(relative, "copy_failed")
            read_issue(relative, "copy_file", error)

    def walk(directory_fd: int, parent: Path) -> None:
        try:
            with os.scandir(directory_fd) as scan:
                entries = sorted(scan, key=lambda entry: entry.name)
        except OSError as error:
            read_issue(parent, "scan_directory", error)
            return
        for entry in entries:
            relative = parent / entry.name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                omit(relative, "entry_unavailable_before_selection")
                read_issue(relative, "stat_without_following_links", error)
                continue
            if stat.S_ISLNK(metadata.st_mode):
                omit(relative, "symlink_not_followed")
            elif stat.S_ISDIR(metadata.st_mode):
                try:
                    child_fd = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                except OSError as error:
                    omit(relative, "directory_unavailable_or_replaced")
                    read_issue(
                        relative, "open_directory_without_following_links", error
                    )
                    continue
                try:
                    walk(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif not stat.S_ISREG(metadata.st_mode):
                omit(relative, "not_a_regular_file")
            elif relative == Path(MANIFEST):
                omit(relative, "reserved_staging_manifest_name", metadata.st_size)
            elif relative.parts[0] in UNIT_PARTITIONS:
                if relative.suffix.lower() not in DIAGNOSTIC_SUFFIXES:
                    omit(relative, "reproducible_unit_fixture", metadata.st_size)
                elif metadata.st_size > DIAGNOSTIC_LIMIT:
                    omit(
                        relative, "unit_diagnostic_exceeds_size_limit", metadata.st_size
                    )
                else:
                    copy_record(directory_fd, relative)
            else:
                copy_record(directory_fd, relative)

    try:
        source_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        read_issue(Path("."), "open_source_without_following_links", error)
        source_missing = error.errno == errno.ENOENT
    else:
        source_missing = False
        try:
            walk(source_fd, Path("."))
        finally:
            os.close(source_fd)

    paths = {record["path"] for record in included}
    reasons = {}
    for record in omitted:
        reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
    manifest = {
        "schema_version": 1,
        "source": str(source),
        "collection_status": "STAGED_WITH_READ_ISSUES" if issues else "STAGED",
        "source_missing": source_missing,
        "policy": {
            "top_level_regular_files": "Retained without size or suffix filtering.",
            "runtime_and_other_regular_files": (
                "Retained recursively, including payloads, checkpoints "
                "and replay evidence."
            ),
            "unit_partitions": sorted(UNIT_PARTITIONS),
            "unit_diagnostic_suffixes": sorted(DIAGNOSTIC_SUFFIXES),
            "unit_diagnostic_limit_bytes": DIAGNOSTIC_LIMIT,
            "symlinks_and_special_files": "Never followed or copied.",
            "test_outcome": "Not evaluated by staging; consult pytest and job results.",
        },
        "included": included,
        "omitted": omitted,
        "read_issues": issues,
        "missing_expected_records": [
            name for name in EXPECTED_RECORDS if name not in paths
        ],
        "summary": {
            "included_files": len(included),
            "included_bytes": sum(record["bytes"] for record in included),
            "omitted_entries": len(omitted),
            "omitted_regular_bytes": sum(record.get("bytes", 0) for record in omitted),
            "omitted_reasons": reasons,
        },
    }
    (destination / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    """Stage records and fail for read issues or required evidence gaps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("ci-artifacts"))
    parser.add_argument("--destination", type=Path, default=Path("ci-artifacts-upload"))
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require every expected record after successful qualification.",
    )
    arguments = parser.parse_args()
    manifest = stage_artifacts(arguments.source, arguments.destination)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "collection_status",
                    "source_missing",
                    "summary",
                    "missing_expected_records",
                    "read_issues",
                )
            },
            indent=2,
        )
    )
    return int(
        bool(manifest["read_issues"])
        or (arguments.require_complete and bool(manifest["missing_expected_records"]))
    )


if __name__ == "__main__":
    raise SystemExit(main())
