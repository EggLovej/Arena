import copy
import sys
from dataclasses import replace

import pytest

from arena.cli import main
from arena.live import LiveMatch
from arena.replay import is_current, read_replay, verify, write_replay
from arena.runner import play_match


def test_a_match_plays_to_the_end_and_its_replay_checks_out(race):
    record = play_match(race, "short", {"red": "steady", "blue": "sprinter"}, seed=3)
    assert record.outcome.reason == "finish"
    assert record.outcome.ranking == (("blue",), ("red",))
    replay = record.replay
    assert replay["frames"]["count"] == record.outcome.ticks + 1
    assert replay["seats"] == [{"seat": "red", "bot": "steady"}, {"seat": "blue", "bot": "sprinter"}]
    assert verify(race, replay) == []


def test_the_same_seed_plays_the_same_match(race):
    lineup = {"red": "dice", "blue": "dice"}
    first = play_match(race, "short", lineup, seed=11).replay
    again = play_match(race, "short", lineup, seed=11).replay
    other = play_match(race, "short", lineup, seed=12).replay
    assert first["actions"] == again["actions"] and first["hashes"] == again["hashes"]
    assert first["actions"] != other["actions"]


@pytest.mark.parametrize(
    "bot, limits, problem",
    [
        ("broken", {}, "this bot always fails"),  # raises
        ("garbage", {}, "not an action"),  # returns something that is not an action
        ("cheater", {}, None),  # returns an action the rules refuse
        ("missing", {}, "load: ModuleNotFoundError"),  # cannot be loaded
        ("sleeper", {"move_time_limit": 0.03}, "more than 0.03s on one move"),
        ("sleeper", {"match_time_limit": 0.25}, "used up its 0.25s for the match"),
    ],
)
def test_a_misbehaving_bot_loses_its_moves_not_the_match(race, monkeypatch, bot, limits, problem):
    monkeypatch.setattr(race, "info", replace(race.info, **limits))
    record = play_match(race, "short", {"red": bot, "blue": "steady"}, seed=6)
    red = record.seats["red"]
    if problem:
        assert problem in red.first_error
    else:
        assert red.errors == 0 and red.refused == red.moves
    for limit in limits.values():  # stopped at its limit, not left to finish thinking
        assert red.think_seconds < limit + 0.04
    assert record.outcome.ranking == (("blue",), ("red",))
    assert verify(race, record.replay) == []


def watched(game, map_id, lineup, seed):
    """The outcome of a match the website plays with bots in every seat."""

    live = LiveMatch(game, map_id, lineup, seed)
    while not live.finished():
        update = live.step()
    return update["outcome"]


def tournament(game, map_id, lineup, seed):
    return play_match(game, map_id, lineup, seed, record_replay=False).outcome.to_json()


@pytest.mark.parametrize("play", [tournament, watched])
def test_a_map_can_have_a_longer_time_limit_than_its_game(race, monkeypatch, play):
    tracks = sys.modules[type(race).__module__].MAPS
    track = tracks["long"]
    monkeypatch.setitem(tracks, "long", replace(track, info=replace(track.info, max_ticks=55)))
    outcome = play(race, "long", {"red": "broken", "blue": "garbage", "green": "cheater"}, 2)
    assert outcome["reason"] == "time" and outcome["ticks"] == 55 > race.info.max_ticks


def test_the_match_plays_each_action_as_its_replay_records_it(race, monkeypatch):
    monkeypatch.setattr(race, "action_from_json", lambda data: 1)
    record = play_match(race, "short", {"red": "sprinter", "blue": "sprinter"}, seed=4)
    assert record.outcome.ticks >= 10  # one step a tick, as decoded, not the three asked for
    assert verify(race, record.replay) == []


def tamper_action(replay):
    replay["actions"] = [[tick, seat, 1 if seat == "blue" else action] for tick, seat, action in replay["actions"]]
    return "final state differs"


def tamper_keyframe(replay):
    replay["frames"]["keyframes"]["2"]["runners"]["red"] = [99]
    return "snapshot differs at tick 2"


def tamper_delta(replay):
    replay["frames"]["deltas"][0]["runners"] = {"red": [999]}
    return "snapshot differs at tick 1"


def drop_delta(replay):
    del replay["frames"]["deltas"][-1]
    return "a delta for every tick"


@pytest.mark.parametrize("tamper", [tamper_action, tamper_keyframe, tamper_delta, drop_delta])
def test_a_tampered_replay_is_caught(race, tamper):
    replay = copy.deepcopy(play_match(race, "long", {"red": "steady", "blue": "sprinter", "green": "steady"}, seed=5, keyframe_every=2).replay)
    expected = tamper(replay)
    assert any(expected in problem for problem in verify(race, replay))


def test_verify_tells_a_broken_replay_from_one_older_rules_changed(race, tmp_path, capsys):
    replay = play_match(race, "short", {"red": "steady", "blue": "sprinter"}, seed=3).replay
    tampered = copy.deepcopy(replay)
    tampered["actions"] = [[tick, seat, 1] for tick, seat, _ in tampered["actions"]]
    broken = write_replay(tmp_path / "broken.json", tampered)
    old = write_replay(tmp_path / "old.json", {**tampered, "fingerprint": "race-0"})
    good = write_replay(tmp_path / "good.json", {**replay, "fingerprint": "race-0"})
    assert not is_current(race, read_replay(old)) and is_current(race, read_replay(broken))

    assert main(["verify", str(old), str(good)]) == 0
    assert "1 of 2 replays play back exactly" in capsys.readouterr().out
    assert main(["verify", str(broken), str(old)]) == 1
    out = capsys.readouterr().out
    assert "broken.json: snapshot differs at tick 1;" in out and "final state differs" in out and "old.json" not in out
