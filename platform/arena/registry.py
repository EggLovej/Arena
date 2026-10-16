"""Find and load the games in ``games/``.

A game is a folder ``games/<id>/`` with an ``arena_game.py`` that defines
``GAME``. Loading a game may change ``sys.path`` and the working directory
(Graphite's does), so load one game per process.
"""

from __future__ import annotations

import importlib.util
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
