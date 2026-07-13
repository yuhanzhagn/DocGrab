from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath


def normalize_relative_file_path(value: str) -> str:
    """Return a portable relative file identity or reject an unsafe path."""
    raw_path = value.strip()
    windows_path = PureWindowsPath(raw_path)
    file_path = raw_path.replace("\\", "/")
    posix_path = PurePosixPath(file_path)
    path_parts = posix_path.parts
    if (
        not file_path
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or posix_path.is_absolute()
        or posix_path.as_posix() == "."
        or ".." in path_parts
    ):
        raise ValueError("file_path must be a non-empty relative path")
    return posix_path.as_posix()
