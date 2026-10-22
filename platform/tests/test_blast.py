"""Blast's own promises beyond what every game makes (test_games.py)."""

import pytest

from arena.live import PERSON, LiveMatch
from arena.registry import load_game

GAME = load_game("blast")  # puts Blast's folder on sys.path, for the imports below

from blast import Bomb, Field  # noqa: E402
from blast.rules import CLOSING, MAX_TICKS, STEPS, Layout, layouts  # noqa: E402

# A corridor along the top and one along the bottom, joined by every other column.
ROWS = (
    "#########",
    "#1......#",
    "#.#.#.#.#",
    "#......2#",
    "#########",
)


def field(crates=()) -> Field:
    field = Field(Layout("test", "Test", "", "", ROWS), 0)
    field.crates = dict(crates)
    return field


def test_a_blast_stops_at_walls_and_the_first_crate_and_sets_off_the_bombs_it_reaches():
    blast = field({(5, 1): "fire", (6, 1): None})
    blast.bombs = {
        (3, 1): Bomb(0, (3, 1), "red", 1, 2),
        (3, 3): Bomb(1, (3, 3), "blue", 99, 1),
    }
    assert blast.observe("red").crates == {(5, 1), (6, 1)} and not blast.observe("red").powers  # what crates hide is hidden
    blast.step({})
    burned = {(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (3, 2), (3, 3), (2, 3), (4, 3)}
    assert blast.flames.keys() == burned and not blast.bombs
    assert blast.crates == {(6, 1): None} and blast.powers == {(5, 1): "fire"}
    # Red stood in the flames; blue, at the far end of the bottom corridor, did not.
    assert blast.bombers["red"].out == 1 and blast.bombers["blue"].out == -1


def test_walls_crates_bombs_and_bombers_are_in_the_way_but_a_bomber_walks_off_its_own_bomb():
    walk = field({(1, 2): None})
    for action in ("up", "left", "down"):
        assert "red" in walk.step({"red": action})
    assert walk.step({"red": "bomb"}) == {}
    assert walk.step({"red": "right"}) == {}
    while walk.bombers["red"].ready > walk.tick:
        assert walk.step({"red": "left"}) == {"red": "it is still stepping"}
    assert walk.step({"red": "left"}) == {"red": "a bomb is in the way"}
    walk.bombers["blue"].tile = (3, 1)
    assert walk.step({"red": "right"}) == {"red": "another bomber is in the way"}
    assert walk.bombers["red"].tile == (2, 1)


def test_a_power_up_counts_once_it_is_picked_up_and_burns_in_the_open():
    pick = field()
    pick.powers = {(2, 1): "speed", (3, 1): "bomb", (4, 1): "fire", (6, 3): "fire"}
    for _ in range(3):
        while pick.step({"red": "right"}):  # until the step is taken
            pass
    red = pick.bombers["red"]
    assert (red.speed, red.bombs, red.range) == (1, 2, 3) and red.ready - red.stepped == STEPS[1]
    pick.bombs[(5, 3)] = Bomb(0, (5, 3), "blue", pick.tick + 1, 1)
    pick.step({})
    assert pick.powers == {}


def test_the_walls_close_in_and_the_last_one_standing_wins():
    match = GAME.new_match("yard", 0)
    while not match.finished():
        match.step({})
    outcome = match.outcome()
    # Red's corner is the first the spiral reaches.
    assert outcome.ranking == (("blue",), ("red",)) and outcome.reason == "elimination"
    assert CLOSING <= outcome.ticks < MAX_TICKS
    # Bombers caught at the same moment share their place.
    match.field.bombers["blue"].out = match.field.bombers["red"].out
    assert match.outcome().ranking == (("red", "blue"),)


@pytest.mark.parametrize("layout", layouts().values(), ids=layouts().keys())
def test_crates_are_as_symmetric_as_the_map(layout):
    rows = [row.translate(str.maketrans("234", "111")) for row in layout.rows]  # a start is a start
    width, height = len(rows[0]), len(rows)
    flips = {"x": [(True, False)], "y": [(False, True)], "xy": [(True, False), (False, True), (True, True)]}[layout.mirror]
    # The mirror images that are the map itself, carrying starts onto each other.
    same = [(x, y) for x, y in flips if [row[::-1] if x else row for row in (rows[::-1] if y else rows)] == rows]
    assert same
    for seed in range(5):
        crates = Field(layout, seed).crates
        for x, y in same:
            assert {(width - 1 - tx if x else tx, height - 1 - ty if y else ty): power for (tx, ty), power in crates.items()} == crates


def test_a_person_walks_and_drops_bombs_from_the_browser():
    live = LiveMatch(GAME, "yard", {"red": PERSON, "blue": "rookie_ai"}, seed=1)
    live.start()
    red = live.match.field.bombers["red"]
    live.match.field.crates.clear()
    # Up is the first way it tries, but the wall above stops it, so it goes right instead.
    assert live.command("red", {"walk": ["up", "right"]}) is None
    for _ in range(4 * STEPS[0]):
        assert "red" not in live.step()["refused"]
    assert red.tile == (5, 1)
    assert live.command("red", "bomb") is None
    live.step()
    assert [bomb.owner for bomb in live.match.field.bombs.values() if bomb.tile == red.tile] == ["red"]
