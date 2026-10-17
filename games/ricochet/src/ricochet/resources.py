"""Locate contributor folders (``ais/``, ``maps/``) and bundled package data."""

from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent


def contributor_directory(name: str) -> Path:
    """The working directory's copy first, then the source checkout's, then the wheel's."""
    working_copy = Path.cwd() / name
    if working_copy.is_dir():
        return working_copy
    source_copy = _PACKAGE.parents[1] / name
    if source_copy.is_dir():
        return source_copy
    return _PACKAGE / "data" / name


def package_data(*parts: str) -> Traversable:
    return files("ricochet").joinpath("data", *parts)
