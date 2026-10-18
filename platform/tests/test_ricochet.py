"""Ricochet's own promises beyond what every game makes (test_games.py)."""

import math

import pytest

from arena.registry import games_directory, load_game
from arena.runner import play_match


@pytest.fixture(scope="module")
def ricochet():
    return load_game("ricochet")


def test_a_shot_is_stored_when_fired_and_when_it_bounces(ricochet):
    match = ricochet.new_match("classic", 0)
    fire = ricochet.action_from_json([0, 0, 0, -1, 1])  # straight up, at the wall above
    match.step({"cyan": fire, "coral": None})
    [(key, first)] = match.snapshot()["shots"].items()
    assert first == ["cyan", 120.0, 306.0, 0.0, -420.0, 1, 0]
    seen = [first]
    while key in match.snapshot()["shots"]:
        shot = match.snapshot()["shots"][key]
        if shot != seen[-1]:
            seen.append(shot)
        match.step({"cyan": None, "coral": None})
    assert [shot[6] for shot in seen] == [0, 1]  # one bounce, then gone at the floor
    bounced = seen[1]
    assert bounced[4] == 420.0 and bounced[2] < 30


def test_a_match_nobody_wins_is_a_draw_on_time(ricochet):
    record = play_match(ricochet, "classic", {"cyan": "reactive_defender_ai", "coral": "reactive_defender_ai"}, record_replay=False)
    assert record.outcome.ranking == (("cyan", "coral"),)
    assert record.outcome.reason == "time"
    assert record.outcome.ticks == 2700


def test_actions_are_recorded_exactly_and_nothing_is_zero(ricochet):
    from ricochet import ControlAction, Vec2
    from ricochet.game.types import validate_action

    action = ControlAction(Vec2(0.123456789, -0.5), Vec2(3, 4), fire=True, shield=False)
    encoded = ricochet.action_to_json(action)
    assert encoded[:2] == [0.123456789, -0.5] and encoded[4] == 1
    assert ricochet.action_from_json(encoded) == validate_action(action)  # aim limited to length one
    assert ricochet.action_to_json(ControlAction(shield=True)) == [0, 0, 0, 0, 2]
    assert ricochet.action_to_json(ControlAction()) == 0
    assert ricochet.action_from_json(0) == ControlAction()
    played = ricochet.action_from_json([0.5, -1, 0.6, 0.8, 3])
    assert played.fire and played.shield and played.move == Vec2(0.5, -1.0)


def test_the_arena_plays_like_ricochets_own_runtime(ricochet, monkeypatch):
    from ricochet.runtime.match import run_headless

    monkeypatch.chdir(games_directory() / "ricochet")  # Ricochet's runtime reads maps/ and ais/ from here
    native = run_headless("sentinel_ai", "reactive_defender_ai", "classic")
    record = play_match(ricochet, "classic", {"cyan": "sentinel_ai", "coral": "reactive_defender_ai"}, record_replay=False)
    assert native.winner == 0 and record.outcome.ranking == (("cyan",), ("coral",))
    assert record.outcome.ticks == native.ticks


@pytest.mark.parametrize("kind", ["tuple", "nan", "flag"])
def test_an_action_of_the_wrong_types_is_refused(ricochet, kind):
    from ricochet import ControlAction, Vec2

    wrong = {
        "tuple": ControlAction(move=(1, 0)),
        "nan": ControlAction(aim=Vec2(math.nan, 0)),
        "flag": ControlAction(fire="yes"),
    }[kind]
    with pytest.raises((TypeError, ValueError)):
        ricochet.action_to_json(wrong)


def test_a_person_moves_aims_and_fires_in_the_browser(ricochet):
    from arena.live import PERSON, LiveMatch

    live = LiveMatch(ricochet, "classic", {"cyan": PERSON, "coral": "direct_ai"}, seed=1)
    start = live.start()
    x, y, _ = start["snapshot"]["tanks"]["cyan"]
    # Held keys stay held: move right, aim straight down from wherever the tank is, fire.
    assert live.command("cyan", {"move": [1, 0], "target": [x, y + 500], "fire": True, "shield": False}) is None
    for _ in range(10):
        update = live.step()
    tanks = live.match.snapshot()["tanks"]
    assert tanks["cyan"][0] > x and tanks["cyan"][1] == y
    # Aiming at the point, from where the tank has moved to (the aim is a tick behind the move).
    assert tanks["cyan"][2] == pytest.approx(math.degrees(math.atan2(500, x - tanks["cyan"][0])), abs=1.5)
    assert any(shot[0] == "cyan" for shot in live.match.snapshot()["shots"].values())
    assert update["refused"] == {} and update["errors"] == {}
    # Letting go stops it.
    live.command("cyan", {"move": [0, 0], "target": None, "fire": False, "shield": False})
    live.step()
    before = live.match.snapshot()["tanks"]["cyan"]
    for _ in range(5):
        live.step()
    assert live.match.snapshot()["tanks"]["cyan"][:2] == before[:2]


@pytest.mark.parametrize(
    "command",
    [
        {"move": [1, 0]},
        {"move": [1, 0], "target": [math.nan, 0], "fire": False, "shield": False},
        {"move": [1, 0], "target": None, "fire": 1, "shield": False},
    ],
)
def test_a_person_s_wrong_command_is_refused(ricochet, command):
    from ricochet import Vec2

    from arena.live import PERSON, LiveMatch

    live = LiveMatch(ricochet, "classic", {"cyan": PERSON, "coral": "direct_ai"}, seed=1)
    assert live.command("cyan", command)  # why it was refused
    assert live.people["cyan"].held["move"] == Vec2(0.0, 0.0)
