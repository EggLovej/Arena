import copy

from arena.replay import read_replay, verify, write_replay
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


def test_bots_that_fail_lose_their_moves_not_the_match(race):
    record = play_match(race, "long", {"red": "broken", "blue": "garbage", "green": "cheater"}, seed=2)
    red, blue, green = (record.seats[seat] for seat in ("red", "blue", "green"))
    assert red.errors == red.moves and "always fails" in red.first_error
    assert blue.errors == blue.moves and "not an action" in blue.first_error
    assert green.errors == 0 and green.refused == green.moves
    assert record.outcome.reason == "time" and record.outcome.ticks == race.info.max_ticks
    assert record.outcome.ranking == (("red", "blue", "green"),)
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
