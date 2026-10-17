"""Ricochet through the Arena contract: the adapter, replays and restoring a match."""

import math

import pytest

from arena.contract import BotContext, state_hash
from arena.registry import games_directory, load_game
from arena.replay import verify
from arena.runner import play_match


@pytest.fixture(scope="module")
def ricochet():
    return load_game("ricochet")


def test_maps_bots_and_seats(ricochet):
    maps = {map_info.id: map_info for map_info in ricochet.maps()}
    assert {"classic", "channels", "crossfire", "generated-3"} <= set(maps)
    assert all(map_info.seats == ("cyan", "coral") for map_info in maps.values())
    assert maps["generated-3"].name == "Generated 3"
    bots = {bot.id: bot for bot in ricochet.bots()}
    assert set(bots) == {"bank_shot_ai", "close_range_ai", "direct_ai", "random_ai", "reactive_defender_ai", "sentinel_ai"}
    assert bots["bank_shot_ai"].name == "Bank Shot"
    assert all(bot.description and bot.kind == "scripted" for bot in bots.values())
    with pytest.raises(ValueError):
        ricochet.load_bot("template_ai")  # where a new bot starts, not a player


def test_maps_and_bots_come_from_its_folder_wherever_it_runs(ricochet, tmp_path, monkeypatch):
    # Graphite's adapter moves into Graphite's folder, which has an ais/ of its own.
    (tmp_path / "ais").mkdir()
    (tmp_path / "ais" / "direct_ai.py").write_text("raise RuntimeError('not a Ricochet bot')\n")
    monkeypatch.chdir(tmp_path)
    assert {bot.id for bot in ricochet.bots()} >= {"direct_ai", "sentinel_ai"}
    bot = ricochet.load_bot("direct_ai")
    bot.reset(BotContext("ricochet", "classic", "cyan", ("cyan", "coral"), 1, 1 / 30, 2700))


def test_a_match_replays_exactly(ricochet):
    record = play_match(ricochet, "crossfire", {"cyan": "sentinel_ai", "coral": "direct_ai"}, seed=3)
    assert sum(report.errors for report in record.seats.values()) == 0
    assert record.outcome.ranking == (("cyan",), ("coral",))
    assert record.outcome.reason == "elimination"
    assert verify(ricochet, record.replay) == []
    frames = record.replay["frames"]
    assert set(frames["keyframes"]["0"]) == {"tanks", "status", "shots"}
    assert any(delta.get("shots") for delta in frames["deltas"])


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


def test_a_restored_match_carries_on_the_same(ricochet):
    match = ricochet.new_match("generated-8", 7)
    bots = {}
    for seat, bot_id in zip(match.seats, ("sentinel_ai", "bank_shot_ai")):
        bots[seat] = ricochet.load_bot(bot_id)
        bots[seat].reset(BotContext("ricochet", "generated-8", seat, match.seats, 1, 1 / 30, 2700))

    def moves(current):
        # As the runner plays them: the action as the replay records it.
        return {seat: ricochet.action_from_json(ricochet.action_to_json(bots[seat].act(current.observe(seat)))) for seat in current.acting()}

    for _ in range(60):
        match.step(moves(match))
    copy = ricochet.restore_match("generated-8", match.save_state())
    assert state_hash(copy.save_state()) == state_hash(match.save_state())
    assert copy.snapshot() == match.snapshot()
    for _ in range(60):
        if match.finished():
            break
        actions = moves(match)
        match.step(actions)
        copy.step(actions)
        assert state_hash(copy.save_state()) == state_hash(match.save_state())


def test_a_match_nobody_wins_is_a_draw_on_time(ricochet):
    record = play_match(ricochet, "classic", {"cyan": "reactive_defender_ai", "coral": "reactive_defender_ai"}, record_replay=False)
    assert record.outcome.ranking == (("cyan", "coral"),)
    assert record.outcome.reason == "time"
    assert record.outcome.ticks == 2700


def test_actions_are_recorded_exactly_and_nothing_is_zero(ricochet):
    from ricochet import ControlAction, Vec2
    from ricochet.api import validate_action

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


@pytest.mark.parametrize("kind", ["text", "dict", "tuple", "nan", "flag"])
def test_anything_but_an_action_is_refused(ricochet, kind):
    from ricochet import ControlAction, Vec2

    wrong = {
        "text": "fire",
        "dict": {"move": [1, 0]},
        "tuple": ControlAction(move=(1, 0)),
        "nan": ControlAction(aim=Vec2(math.nan, 0)),
        "flag": ControlAction(fire="yes"),
    }[kind]
    with pytest.raises((TypeError, ValueError)):
        ricochet.action_to_json(wrong)
