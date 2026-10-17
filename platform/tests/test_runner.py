import copy
import sys
from dataclasses import replace

from arena.cli import main
from arena.replay import is_current, read_replay, verify, write_replay
from arena.runner import play_match


def test_a_match_plays_to_the_end_and_its_replay_checks_out(race):
    record = play_match(race, "short", {"red": "steady", "blue": "sprinter"}, seed=3)
    assert record.outcome.reason == "finish"
    assert record.outcome.winner in ("red", "blue")
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


def test_a_changed_action_is_caught(race):
    replay = play_match(race, "long", {"red": "steady", "blue": "sprinter", "green": "steady"}, seed=5).replay
    tampered = copy.deepcopy(replay)
    tampered["actions"] = [[tick, seat, 1 if seat == "blue" else action] for tick, seat, action in tampered["actions"]]
    problems = verify(race, tampered)
    assert any("final state differs" in problem for problem in problems)


def test_a_changed_frame_is_caught(race):
    replay = play_match(race, "short", {"red": "steady", "blue": "steady"}, seed=1, keyframe_every=2).replay
    tampered = copy.deepcopy(replay)
    tampered["frames"]["keyframes"]["2"]["runners"]["red"] = [99]
    assert "snapshot differs at tick 2" in verify(race, tampered)


def test_a_changed_delta_is_caught(race):
    replay = play_match(race, "short", {"red": "steady", "blue": "steady"}, seed=1).replay
    tampered = copy.deepcopy(replay)
    tampered["frames"]["deltas"][0]["runners"] = {"red": [999]}
    assert verify(race, tampered) == ["snapshot differs at tick 1"]
    del tampered["frames"]["deltas"][-1]
    assert "a delta for every tick" in verify(race, tampered)[0]


def test_bots_that_fail_lose_their_moves_not_the_match(race):
    record = play_match(race, "long", {"red": "broken", "blue": "garbage", "green": "cheater"}, seed=2)
    red, blue, green = (record.seats[seat] for seat in ("red", "blue", "green"))
    assert red.errors == red.moves and "always fails" in red.first_error
    assert blue.errors == blue.moves and "not an action" in blue.first_error
    assert green.errors == 0 and green.refused == green.moves
    assert record.outcome.reason == "time" and record.outcome.ticks == race.info.max_ticks
    assert record.outcome.ranking == (("red", "blue", "green"),)
    assert verify(race, record.replay) == []


def test_a_map_can_have_a_longer_time_limit_than_its_game(race, monkeypatch):
    tracks = sys.modules[type(race).__module__].MAPS
    track = tracks["long"]
    monkeypatch.setitem(tracks, "long", replace(track, info=replace(track.info, max_ticks=55)))
    record = play_match(race, "long", {"red": "broken", "blue": "garbage", "green": "cheater"}, seed=2)
    assert record.outcome.reason == "time" and record.outcome.ticks == 55 > race.info.max_ticks
    assert record.replay["max_ticks"] == 55
    assert verify(race, record.replay) == []


def test_replays_round_trip_plain_and_gzipped(race, tmp_path):
    replay = play_match(race, "short", {"red": "steady", "blue": "dice"}, seed=8).replay
    for name in ("match.json", "match.json.gz"):
        path = write_replay(tmp_path / name, replay)
        assert read_replay(path) == replay


def test_a_summary_row_has_what_the_tables_need(race):
    row = play_match(race, "short", {"red": "steady", "blue": "sprinter"}, seed=0, match_id="m1", record_replay=False).summary()
    assert row["id"] == "m1" and row["lineup"] == {"red": "steady", "blue": "sprinter"}
    assert set(row["seats"]["red"]) >= {"bot", "moves", "errors", "refused", "think_ms_mean", "think_ms_max"}


def test_a_bot_that_fails_to_load_forfeits_and_the_match_goes_on(race):
    record = play_match(race, "short", {"red": "missing", "blue": "steady"}, seed=6)
    red = record.seats["red"]
    assert red.errors == 1 and red.first_error.startswith("load: ModuleNotFoundError")
    assert red.moves == record.outcome.ticks and record.outcome.winner == "blue"
    assert verify(race, record.replay) == []


def test_a_bot_that_thinks_too_long_on_one_move_forfeits(race, monkeypatch):
    monkeypatch.setattr(race, "info", replace(race.info, move_time_limit=0.03))
    record = play_match(race, "short", {"red": "sleeper", "blue": "steady"}, seed=6)
    red = record.seats["red"]
    assert red.errors == 1 and "more than 0.03s on one move" in red.first_error
    assert red.slowest_seconds < 0.09  # stopped, not left to finish its 0.1s nap
    assert record.outcome.winner == "blue" and not any(seat == "red" for _, seat, _ in record.replay["actions"])


def test_a_bot_that_uses_up_its_time_for_the_match_forfeits(race, monkeypatch):
    monkeypatch.setattr(race, "info", replace(race.info, match_time_limit=0.25))
    record = play_match(race, "long", {"red": "sleeper", "blue": "steady", "green": "steady"}, seed=6)
    red = record.seats["red"]
    assert red.errors == 1 and "used up its 0.25s" in red.first_error
    assert 2 <= sum(seat == "red" for _, seat, _ in record.replay["actions"]) <= 3
    assert red.think_seconds < 0.3


def test_the_match_plays_each_action_as_its_replay_records_it(race, monkeypatch):
    monkeypatch.setattr(race, "action_from_json", lambda data: 1)
    record = play_match(race, "short", {"red": "sprinter", "blue": "sprinter"}, seed=4)
    assert record.outcome.ticks >= 10  # one step a tick, as decoded, not the three asked for
    assert verify(race, record.replay) == []


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
