from pathlib import Path

import pytest

from arena.registry import load_game

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def race(monkeypatch):
    """The toy game in fixtures/games/race, found through the registry like a real game."""

    monkeypatch.setenv("ARENA_ROOT", str(FIXTURES))
    return load_game("race")
