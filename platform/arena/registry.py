"""Find and load the games in ``games/``, and the files each one's matches depend on.

A game is a folder ``games/<id>/`` with an ``arena_game.py`` that defines
``GAME`` (see ``arena.contract``). Loading a game may change ``sys.path`` and the
working directory (Graphite's does), so load one game per process.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from functools import cache
from pathlib import Path

from arena.contract import Game


def repo_root() -> Path:
    """The Arena checkout: ``ARENA_ROOT`` if set, else the folder above ``platform/``."""

    if os.environ.get("ARENA_ROOT"):
        return _resolve(os.environ["ARENA_ROOT"])
    return Path(__file__).resolve().parents[2]


@cache
def _resolve(path: str) -> Path:
    # Once only: a relative ARENA_ROOT means relative to where we started, and loading a
    # game can change the working directory.
    return Path(path).resolve()


def games_directory() -> Path:
    return repo_root() / "games"


def available_games() -> tuple[str, ...]:
    return tuple(sorted(path.parent.name for path in games_directory().glob("*/arena_game.py")))


@cache
def load_game(game_id: str) -> Game:
    path = games_directory() / game_id / "arena_game.py"
    if not path.is_file():
        raise ValueError(f"No game named {game_id!r}. Games: {', '.join(available_games()) or 'none'}")
    spec = importlib.util.spec_from_file_location(f"arena_game_{game_id}", path)
    module = importlib.util.module_from_spec(spec)
    # Registered first, as an import would: dataclasses and pickling look a class's module up here.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[spec.name]
        raise
    game = getattr(module, "GAME", None)
    if game is None:
        raise ValueError(f"{path} must define GAME")
    if game.info.id != game_id:
        raise ValueError(f"{path} says it is {game.info.id!r}, but its folder is {game_id!r}")
    return game


def game_files(game_id: str, *parts: str) -> list[Path]:
    """``arena_game.py`` and the files its ``files.json`` lists under ``parts`` ("rules", "bots"),
    with each listed folder expanded to every file in it."""

    folder = games_directory() / game_id
    listed = json.loads((folder / "files.json").read_text())
    files = {folder / "arena_game.py"}
    for entry in (entry for part in parts for entry in listed[part]):
        path = folder / entry
        if path.is_dir():
            files.update(found for found in path.rglob("*") if found.is_file() and "__pycache__" not in found.parts)
        elif path.is_file():
            files.add(path)
        else:
            raise FileNotFoundError(f"{game_id}'s files.json lists {entry}, which does not exist")
    return sorted(files)


def fingerprint(game_id: str) -> str:
    """A hash of what a game's replays depend on: its rules and maps, and the adapter that records them.
    A replay recorded with another fingerprint may no longer play back, and that is not an error."""

    return hash_files(game_files(game_id, "rules"), games_directory() / game_id)


def hash_files(files: list[Path], root: Path, digest=None) -> str:
    """A short hash of the files' names (relative to ``root``) and contents."""

    digest = digest or hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:16]
