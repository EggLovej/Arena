"""Matches people play on the website (arena.live), with the Race game."""

import pytest

from arena.frames import apply
from arena.live import PERSON, LiveMatch
from arena.runner import play_match


def play_out(live: LiveMatch, each_tick=lambda live: None) -> list[dict]:
    updates = []
    while not live.finished():
        each_tick(live)
        live.think()
        updates.append(live.step())
    return updates


@pytest.mark.parametrize(
    "map_id, lineup, steps",
    [
        ("short", {"red": "sprinter", "blue": "steady"}, 3),
        ("long", {"red": "steady", "blue": "sprinter", "green": "dice"}, 2),
    ],
)
def test_a_person_playing_like_a_bot_gets_the_bots_result(race, map_id, lineup, steps):
    expected = play_match(race, map_id, lineup, seed=5, record_replay=False)
    live = LiveMatch(race, map_id, {**lineup, "red": PERSON}, seed=5)
    start = live.start()
    assert start["people"] == {"red": {"speed": None}}

    shown = start["snapshot"]
    updates = play_out(live, lambda live: live.command("red", steps))
    for update in updates:
        shown = apply(shown, update["delta"])
    assert shown == live.match.snapshot()
    assert updates[-1]["outcome"] == expected.outcome.to_json()
    assert all(update["outcome"] is None for update in updates[:-1])
    assert updates[0]["tick"] == 1 and updates[-1]["tick"] == len(updates)


def test_commands_are_checked(race):
    live = LiveMatch(race, "short", {"red": PERSON, "blue": "steady"}, seed=1)
    assert "not a number of steps" in live.command("red", "fast")
    assert "not played by a person" in live.command("blue", 2)
    assert live.command("red", 7) is None  # a number of steps, but one the rules refuse
    assert live.step()["refused"] == {"red": "7 is not 1, 2 or 3"}


def test_only_the_seats_whose_turn_it_is_choose_and_play(race):
    live = LiveMatch(race, "short", {"red": PERSON, "blue": "steady"}, seed=1)
    match = live.match
    match.acting = lambda: (match.seats[match.tick % 2],)  # Race made turn-based: red, blue, red, ...
    played = []
    step = match.step

    def taking_turns(actions):
        played.append(dict(actions))
        return step({seat: actions.get(seat) for seat in match.seats})

    match.step = taking_turns
    live.command("red", 1)
    for _ in range(4):
        live.think()
        live.step()
    assert played == [{"red": 1}, {"blue": 2}, {"red": 1}, {"blue": 2}]
    assert live.reports["blue"].moves == 2


def test_bots_think_once_a_tick(race):
    live = LiveMatch(race, "long", {"red": "dice", "blue": PERSON, "green": "steady"}, seed=1)
    live.think()
    live.think()  # already decided for this tick: nothing changes
    live.step()
    assert live.reports["red"].moves == live.reports["green"].moves == 1


def test_bots_that_break_are_reported_and_the_match_goes_on(race):
    live = LiveMatch(race, "long", {"red": PERSON, "blue": "broken", "green": "missing"}, seed=2)
    updates = play_out(live, lambda live: live.command("red", 1))
    errors = updates[-1]["errors"]
    assert "this bot always fails" in errors["blue"]
    assert errors["green"].startswith("load: ")
    assert updates[-1]["outcome"]["ranking"][0] == ["red"]


def test_a_lineup_must_fill_the_map_with_bots_or_people_the_game_takes(race, monkeypatch):
    with pytest.raises(ValueError):
        LiveMatch(race, "short", {"red": PERSON}, seed=0)
    with pytest.raises(ValueError):
        LiveMatch(race, "short", {"red": PERSON, "blue": "nobody"}, seed=0)
    monkeypatch.delattr(type(race), "person")
    with pytest.raises(ValueError, match="cannot be played by people"):
        LiveMatch(race, "short", {"red": PERSON, "blue": "steady"}, seed=0)
