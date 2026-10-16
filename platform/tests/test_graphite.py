"""Graphite through the Arena contract: the adapter, replays and restoring a match."""

import pytest

from arena.contract import BotContext, state_hash
from arena.registry import load_game
from arena.replay import verify
from arena.runner import play_match


@pytest.fixture(scope="module")
def graphite():
    return load_game("graphite")


def test_maps_bots_and_seats(graphite):
    maps = {map_info.id: map_info for map_info in graphite.maps()}
    assert maps["classic"].seats == ("red", "blue")
    assert maps["triangle"].seats == ("red", "blue", "green")
    bots = {bot.id: bot for bot in graphite.bots()}
    assert {"strategic_ai", "careful_ai", "aggressive_ai", "neural_ai", "beast_ai", "atlas_ai"} <= set(bots)
    assert bots["neural_ai"].kind == "neural" and bots["strategic_ai"].kind == "scripted"
    assert bots["atlas_ai"].seat_counts == (2,)
    assert all(bot.description for bot in bots.values())


def test_a_match_replays_exactly(graphite):
    record = play_match(graphite, "triangle", {"red": "strategic_ai", "blue": "careful_ai", "green": "aggressive_ai"}, seed=4)
    assert sum(report.errors for report in record.seats.values()) == 0
    assert len(sum(record.outcome.ranking, ())) == 3
    assert verify(graphite, record.replay) == []
    snapshot = record.replay["frames"]["keyframes"]["0"]
    assert set(snapshot) == {"bases", "fleets"}


def test_a_restored_match_carries_on_the_same(graphite):
    match = graphite.new_match("classic", 7)
    bots = {}
    for seat, bot_id in zip(match.seats, ("strategic_ai", "careful_ai")):
        bots[seat] = graphite.load_bot(bot_id)
        bots[seat].reset(BotContext("graphite", "classic", seat, match.seats, 1, 0.1, 1800))

    def moves(current):
        return {seat: bots[seat].act(current.observe(seat)) for seat in current.acting()}

    for _ in range(150):
        match.step(moves(match))
    copy = graphite.restore_match("classic", match.save_state())
    assert state_hash(copy.save_state()) == state_hash(match.save_state())
    assert copy.snapshot() == match.snapshot()
    for _ in range(150):
        actions = moves(match)
        match.step(actions)
        copy.step({seat: graphite.action_from_json(graphite.action_to_json(action)) if action is not None else None for seat, action in actions.items()})
        assert state_hash(copy.save_state()) == state_hash(match.save_state())


@pytest.mark.parametrize("bot", ["neural_ai", "beast_ai"])
def test_the_neural_bots_play_their_committed_networks(graphite, bot):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    record = play_match(graphite, "classic", {"red": bot, "blue": "strategic_ai"}, seed=2)
    assert record.seats["red"].errors == 0
    assert sum(seat == "red" for _, seat, _ in record.replay["actions"]) > 10
