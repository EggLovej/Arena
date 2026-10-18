import sys
from dataclasses import replace

import pytest

from arena.leaderboard import points, standings
from arena.contract import Outcome
from arena.tournament import run, schedule


def test_every_group_plays_every_seating_and_shows_one(race):
    plans = schedule(race)
    short = [plan for plan in plans if plan.map == "short"]
    long = [plan for plan in plans if plan.map == "long"]
    assert len(short) == 3 * 2  # 3 pairs of steady, sprinter, dice; both seatings
    assert len(long) == 0  # only two bots play 3-player matches (dice is 2-player only)
    assert sum(plan.showcase for plan in short) == 3
    assert len({plan.match_id for plan in plans}) == len(plans)
    assert all("broken" not in plan.bots for plan in plans)


def test_chosen_bots_and_rounds(race):
    plans = schedule(race, bots=["steady", "sprinter", "broken"], maps=["long"], rounds=2)
    assert len(plans) == 6 * 2
    assert sum(plan.showcase for plan in plans) == 1
    with pytest.raises(ValueError, match="no bot"):
        schedule(race, bots=["nobody"])
    with pytest.raises(ValueError, match="no map"):
        schedule(race, maps=["moon"])


def test_a_bot_can_sit_out_a_group_of_maps(race, monkeypatch):
    module = sys.modules[type(race).__module__]
    track = module.MAPS["short"]
    monkeypatch.setitem(module.MAPS, "short", replace(track, info=replace(track.info, group="Large maps")))
    info, bot = module.BOTS["sprinter"]
    monkeypatch.setitem(module.BOTS, "sprinter", (replace(info, sits_out=("Large maps",)), bot))
    short = [plan for plan in schedule(race) if plan.map == "short"]
    assert len(short) == 2  # steady and dice only
    assert all("sprinter" not in plan.bots for plan in short)


@pytest.mark.parametrize("workers", [1, 2])
def test_a_tournament_runs_in_one_or_several_processes(race, workers):
    plans = schedule(race)
    finished = list(run("race", plans, workers))
    assert sorted(plan.match_id for plan, _, _ in finished) == sorted(plan.match_id for plan in plans)
    for plan, row, replay in finished:
        assert row["id"] == plan.match_id
        assert (replay is not None) == plan.showcase


def test_standings_make_one_table_per_match_size(race):
    rows = [row for _, row, _ in run("race", schedule(race, bots=["steady", "sprinter", "broken"]), 1)]
    tables = standings(race, rows)
    assert sorted(tables) == ["2", "3"]
    two = tables["2"]
    assert two["matches"] == 6 and two["maps"] == ["short"]
    assert {entry["bot"] for entry in two["entries"]} == {"steady", "sprinter", "broken"}
    assert [entry["rating"] for entry in two["entries"]] == sorted((entry["rating"] for entry in two["entries"]), reverse=True)
    steady = next(entry for entry in two["entries"] if entry["bot"] == "steady")
    assert steady["rating"] == 1000  # the anchor
    assert steady["games"] == steady["wins"] + steady["draws"] + steady["losses"] == 4
    broken = next(entry for entry in two["entries"] if entry["bot"] == "broken")
    assert broken["errors"] > 0
    head = two["head_to_head"]
    for a in head:
        for b, cell in head[a].items():
            assert cell["games"] == head[b][a]["games"]
            assert cell["points"] + head[b][a]["points"] == pytest.approx(cell["games"])
    # 1 point for first place and 0 for last; seats that share a place share its points.
    assert [points(Outcome((("a",), ("b", "c")), "finish", 10), seat) for seat in "abc"] == [1.0, 0.25, 0.25]


def test_bots_with_equal_ratings_are_listed_by_id(race):
    def row(winner, loser):
        seats = {seat: {"bot": bot, "moves": 1, "errors": 0, "refused": 0, "think_ms_mean": 0.0}
                 for seat, bot in (("red", winner), ("blue", loser))}
        return {"map": "short", "lineup": {"red": winner, "blue": loser}, "seats": seats,
                "outcome": {"ranking": [["red"], ["blue"]], "reason": "finish", "ticks": 5}}

    for names in (["amy", "zed"], ["zed", "amy"], ["kim", "bob"]):
        rows = [row(names[0], names[1]), row(names[1], names[0])]
        assert [entry["bot"] for entry in standings(race, rows)["2"]["entries"]] == sorted(names)
