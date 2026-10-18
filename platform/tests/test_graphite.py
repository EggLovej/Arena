"""Graphite's own promises beyond what every game makes (test_games.py)."""

import pytest

from arena.live import PERSON, LiveMatch
from arena.registry import load_game
from arena.runner import play_match


@pytest.fixture(scope="module")
def graphite():
    return load_game("graphite")


@pytest.mark.parametrize("bot", ["neural_ai", "beast_ai"])
def test_the_neural_bots_play_their_committed_networks_the_same_way_twice(graphite, bot):
    # The networks draw their moves by chance, from the match's seed: a seeded match can be played again.
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    first, second = (play_match(graphite, "classic", {"red": bot, "blue": "strategic_ai"}, seed=2) for _ in range(2))
    assert first.seats["red"].errors == 0
    assert sum(seat == "red" for _, seat, _ in first.replay["actions"]) > 10
    assert first.replay["actions"] == second.replay["actions"]


def test_an_action_must_name_bases_by_id_and_send_whole_units(graphite):
    from game_types import SendAction

    assert graphite.action_to_json(SendAction("north", "center", 3)) == ["north", "center", 3]
    with pytest.raises(TypeError, match="source_id must be a base id"):
        graphite.action_to_json(SendAction(["north"], "center", 3))
    with pytest.raises(TypeError, match="target_id must be a base id"):
        graphite.action_to_json(SendAction("north", 7, 3))
    with pytest.raises(TypeError, match="units must be a whole number"):
        graphite.action_to_json(SendAction("north", "center", 3.0))


def fleets_of(live: LiveMatch, seat: str) -> list:
    return [fleet for fleet in live.match.snapshot()["fleets"].values() if fleet[0] == seat]


def test_a_person_sends_and_streams(graphite):
    live = LiveMatch(graphite, "crossroads", {"red": PERSON, "blue": "careful_ai"}, seed=0)
    board = live.start()["board"]
    bases = {base["id"]: base for base in board["bases"]}
    home = next(base for base, (owner, _) in live.match.snapshot()["bases"].items() if owner == "red")
    route = next(edge for edge in board["edges"] if home in (edge["a"], edge["b"]))
    target = route["b"] if route["a"] == home else route["a"]
    assert target in bases

    for _ in range(30):  # let the home base grow
        live.step()
    units = live.match.snapshot()["bases"][home][1]
    assert live.command("red", {"send": [home, target]}) is None
    update = live.step()
    assert update["refused"] == {}
    (fleet,) = fleets_of(live, "red")
    assert fleet[1] == units // 2 and fleet[2:4] == [home, target]

    assert live.command("red", {"stream": [home, target]}) is None
    assert live.step()["people"] == {"red": {"streams": {home: target}}}
    assert len(fleets_of(live, "red")) == 2
    assert live.command("red", {"stop": home}) is None
    assert live.step()["people"] == {"red": {"streams": {}}}

    assert live.command("red", {"send": home}) is not None
    assert live.command("red", {"fly": [home, target]}) is not None
    assert live.command("red", [home, target, 3]) is not None
