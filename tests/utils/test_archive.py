"""Supported safe links and malformed archive manifests use one explicit policy."""

import io
import stat
import tarfile
import zipfile

import pytest

from plato.utils.archive import UnsafeArchiveError, extract_archive


def write_tar(path, members):
    with tarfile.open(path, "w") as archive:
        for name, type_, value in members:
            info = tarfile.TarInfo(name)
            info.type = type_
            if type_ == tarfile.REGTYPE:
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))
            else:
                info.linkname = value
                archive.addfile(info)


def test_safe_tar_links_and_dot_directory(tmp_path):
    archive = tmp_path / "safe.tar"
    write_tar(
        archive,
        [
            (".", tarfile.DIRTYPE, ""),
            ("nested/data", tarfile.REGTYPE, b"safe"),
            ("nested/sym", tarfile.SYMTYPE, "data"),
            ("hard", tarfile.LNKTYPE, "nested/data"),
        ],
    )
    destination = tmp_path / "result"
    extract_archive(archive, destination)
    assert (destination / "nested/data").read_bytes() == b"safe"
    assert (destination / "nested/sym").read_bytes() == b"safe"
    assert (destination / "hard").read_bytes() == b"safe"


@pytest.mark.parametrize(
    "members",
    [
        [("link", tarfile.SYMTYPE, "dir"), ("link/file", tarfile.REGTYPE, b"data")],
        [("one", tarfile.SYMTYPE, "two"), ("two", tarfile.SYMTYPE, "file")],
        [("link", tarfile.SYMTYPE, "file"), ("link", tarfile.REGTYPE, b"replace")],
        [("pipe", tarfile.FIFOTYPE, "")],
        [("C:/outside/file", tarfile.REGTYPE, b"data")],
        [("..\\outside\\file", tarfile.REGTYPE, b"data")],
    ],
)
def test_unsafe_tar_manifest_is_rejected_before_extraction(tmp_path, members):
    archive = tmp_path / "fixture.tar"
    write_tar(archive, [("safe", tarfile.REGTYPE, b"safe"), *members])
    destination = tmp_path / "result"
    with pytest.raises(UnsafeArchiveError, match="Unsafe"):
        extract_archive(archive, destination)
    assert list(destination.iterdir()) == []


def test_zip_symlink_entry_is_rejected(tmp_path):
    archive = tmp_path / "fixture.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("safe", b"safe")
        zipped.writestr(info, "../outside")
    destination = tmp_path / "result"
    with pytest.raises(UnsafeArchiveError, match="ZIP symlink"):
        extract_archive(archive, destination)
    assert list(destination.iterdir()) == []
