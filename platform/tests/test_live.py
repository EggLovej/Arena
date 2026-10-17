"""Matches people play on the website (arena.live), with the Race game and with Graphite."""

import pytest

from arena.frames import apply
from arena.live import PERSON, LiveMatch, needs
from arena.registry import games_directory, load_game
from arena.runner import play_match


def play_out(live: LiveMatch, each_tick=lambda live: None) -> list[dict]:
    updates = []
    while not live.finished():
        each_tick(live)
        live.think()
        updates.append(live.step())
    return updates


def test_a_person_plays_against_a_bot(race):
    live = LiveMatch(race, "short", {"red": PERSON, "blue": "steady"}, seed=3)
    start = live.start()
    assert start["board"] == {"goal": 10, "seats": ["red", "blue"]}
    assert start["people"] == {"red": {"waiting": 0}}

    shown = start["snapshot"]
    updates = play_out(live, lambda live: live.command("red", 3))
    for update in updates:
        shown = apply(shown, update["delta"])
    assert shown == live.match.snapshot()
    sprinter = play_match(race, "short", {"red": "sprinter", "blue": "steady"}, seed=3, record_replay=False)
    assert updates[-1]["outcome"] == sprinter.outcome.to_json()  # three steps a tick, as Sprinter
    assert all(update["outcome"] is None for update in updates[:-1])
    assert updates[0]["tick"] == 1 and updates[-1]["tick"] == len(updates)


def test_a_person_playing_like_a_bot_gets_the_bots_result(race):
    expected = play_match(race, "long", {"red": "steady", "blue": "sprinter", "green": "dice"}, seed=5, record_replay=False)
    live = LiveMatch(race, "long", {"red": PERSON, "blue": "sprinter", "green": "dice"}, seed=5)
    updates = play_out(live, lambda live: live.command("red", 2))
    assert updates[-1]["outcome"] == expected.outcome.to_json()


def test_commands_are_queued_one_a_tick_and_checked(race):
    live = LiveMatch(race, "short", {"red": PERSON, "blue": "steady"}, seed=1)
    assert live.command("red", 2) is None
    assert live.command("red", 1) is None
    assert live.start()["people"] == {"red": {"waiting": 2}}
    assert "not an action" in live.command("red", "fast")
    assert "not played by a person" in live.command("blue", 2)
    live.step()
    live.step()
    assert live.step()["people"] == {"red": {"waiting": 0}}
    live.command("red", 7)  # an action, but one the rules refuse
    assert live.step()["refused"] == {"red": "7 is not 1, 2 or 3"}


def test_only_the_seats_whose_turn_it_is_choose_and_play(race):
    live = LiveMatch(race, "short", {"red": PERSON, "blue": "steady"}, seed=1)
    match = live.match
    match.acting = lambda: (match.seats[match.tick % 2],)  # Race made turn-based: red, blue, red, ...
    played = []
    step = match.step

    def taking_turns(actions):
        played.append(sorted(actions))
        return step({seat: actions.get(seat) for seat in match.seats})

    match.step = taking_turns
    live.command("red", 2)
    live.command("red", 1)
    waiting = []
    for _ in range(4):
        live.think()
        waiting.append(live.step()["people"]["red"]["waiting"])
    assert played == [["red"], ["blue"], ["red"], ["blue"]]
    assert waiting == [1, 1, 0, 0]  # red's second command waits for red's next turn
    assert live.reports["blue"].moves == 2


def test_bots_that_break_are_reported_and_the_match_goes_on(race):
    live = LiveMatch(race, "long", {"red": PERSON, "blue": "broken", "green": "missing"}, seed=2)
    updates = play_out(live, lambda live: live.command("red", 1))
    errors = updates[-1]["errors"]
    assert "this bot always fails" in errors["blue"]
    assert errors["green"].startswith("load: ")
    assert updates[-1]["outcome"]["ranking"][0] == ["red"]


def test_a_lineup_must_fill_the_map(race):
    with pytest.raises(ValueError):
        LiveMatch(race, "short", {"red": PERSON}, seed=0)
    with pytest.raises(ValueError):
        LiveMatch(race, "short", {"red": PERSON, "blue": "nobody"}, seed=0)


def test_a_game_without_web_needs_needs_nothing(race):
    assert needs(race, ["steady", "dice"]) == {"packages": [], "files": []}


@pytest.fixture(scope="module")
def graphite():
    return load_game("graphite")


def fleets_of(live: LiveMatch, seat: str) -> list:
    return [fleet for fleet in live.match.snapshot()["fleets"].values() if fleet[0] == seat]


def test_a_person_sends_and_streams_in_graphite(graphite):
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


def test_graphite_names_what_its_networks_need_in_the_browser(graphite):
    assert needs(graphite, ["strategic_ai", "careful_ai"]) == {"packages": [], "files": []}
    wanted = needs(graphite, ["neural_ai", "beast_ai", "strategic_ai"])
    assert wanted["packages"] == ["numpy"]
    assert len(wanted["files"]) == 2
    for path in wanted["files"]:
        assert path.endswith(".pt")
        assert (games_directory() / "graphite" / path).is_file()


def test_bots_think_before_the_tick_is_played(graphite):
    live = LiveMatch(graphite, "triangle", {"red": "strategic_ai", "blue": PERSON, "green": "aggressive_ai"}, seed=1)
    live.think()
    planned = live._planned
    assert set(planned) == {"red", "green"}
    live.think()  # already decided for this tick: nothing changes
    assert live._planned is planned
    update = live.step()
    assert update["tick"] == 1 and set(update["thinking_ms"]) == {"red", "green"}
