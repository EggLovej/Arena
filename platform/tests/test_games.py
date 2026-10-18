"""What every game in games/ promises the platform, checked for each one."""

import json
import subprocess
import sys

import pytest

from arena.live import needs
from arena.registry import available_games, game_files, games_directory, load_game
from arena.replay import verify
from arena.runner import play_match

GAMES = available_games()


@pytest.fixture(params=GAMES, scope="module")
def game(request):
    return load_game(request.param)


def test_maps_bots_seats_and_themes(game):
    seats = {seat.id for seat in game.info.seats}
    maps = game.maps()
    assert maps and all(map_info.seats and set(map_info.seats) <= seats for map_info in maps)
    bots = {bot.id: bot for bot in game.bots()}
    assert len(bots) == len(game.bots()) and all(bot.name and bot.description for bot in bots.values())
    anchor = game.info.rating_anchor
    assert anchor is None or bots[anchor[0]].leaderboard
    themes = game.themes()
    assert themes and len({theme["id"] for theme in themes}) == len(themes)
    assert all(theme["name"] and theme["description"] for theme in themes)


def test_a_match_replays_exactly(game):
    anchor = game.info.rating_anchor[0]
    map_info = game.maps()[0]
    record = play_match(game, map_info.id, {seat: anchor for seat in map_info.seats}, seed=4)
    assert not any(report.errors for report in record.seats.values())
    assert verify(game, record.replay) == []


def test_garbage_is_not_an_action(game):
    # Each game's own odd cases are in its test file.
    for garbage in (object(), "fire", [1, 2, 3]):
        with pytest.raises((TypeError, ValueError)):
            game.action_to_json(garbage)


def test_the_browser_gets_every_file_a_bot_needs(game):
    listed = set(game_files(game.info.id, "rules", "bots"))
    wanted = needs(game, [bot.id for bot in game.bots()])
    for name in wanted["files"]:
        assert games_directory() / game.info.id / name in listed


# Plays a tick with every bot, then prints the game's files that Python imported.
TRACE = """
import json, sys
from pathlib import Path
if sys.argv[2] == "browser":
    sys.modules["torch"] = None  # the browser has no PyTorch
from arena.live import LiveMatch
from arena.registry import games_directory, load_game
game = load_game(sys.argv[1])
folder = games_directory() / sys.argv[1]
map_info = game.maps()[0]
for bot in game.bots():
    live = LiveMatch(game, map_info.id, {seat: bot.id for seat in map_info.seats}, seed=1)
    assert not live.step()["errors"], bot.id
loaded = {Path(module.__file__) for module in list(sys.modules.values()) if getattr(module, "__file__", None)}
print(json.dumps(sorted(str(path) for path in loaded if path.is_absolute() and path.is_relative_to(folder))))
"""


@pytest.mark.parametrize("where", ["tournament", "browser"])
def test_every_file_a_match_imports_is_listed(game, where):
    if where == "tournament":
        pytest.importorskip("torch")
    run = subprocess.run([sys.executable, "-c", TRACE, game.info.id, where], capture_output=True, text=True, check=True)
    imported = {games_directory() / game.info.id / path for path in json.loads(run.stdout)}
    assert imported - set(game_files(game.info.id, "rules", "bots")) == set()
