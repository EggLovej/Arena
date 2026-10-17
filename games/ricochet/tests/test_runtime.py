from pathlib import Path

from ricochet.game.config import RulesConfig
from ricochet.game.engine import GameEngine
from ricochet.game.maps import load_map
from ricochet.game.types import ControlAction
from ricochet.runtime.loading import available_ais, load_controller
from ricochet.runtime.match import MatchSession, run_headless
from ricochet.runtime.replay import verify_replay
from ricochet.runtime.tournament import run_tournament, standings, write_results


def test_bundled_ais_are_discovered_and_mirror_instances_are_separate() -> None:
    assert available_ais() == (
        "bank_shot_ai",
        "close_range_ai",
        "direct_ai",
        "random_ai",
        "reactive_defender_ai",
        "sentinel_ai",
        "template_ai",
    )
    assert load_controller("direct_ai") is not load_controller("direct_ai")


def test_controller_faults_forfeit_after_three_errors(tmp_path: Path) -> None:
    ai_dir = tmp_path / "ais"
    ai_dir.mkdir()
    (ai_dir / "broken_ai.py").write_text(
        "class Bot:\n"
        "    def reset(self, context): pass\n"
        "    def act(self, observation): raise RuntimeError('broken')\n"
        "def create_controller(): return Bot()\n"
    )
    controller = load_controller("broken_ai", ai_dir)

    class NoopController:
        def reset(self, context) -> None:
            pass

        def act(self, observation) -> ControlAction:
            return ControlAction()

    engine = GameEngine(load_map("classic"))
    session = MatchSession(engine, (controller, NoopController()), ("broken", "noop"))
    while not session.finished:
        session.step()
    summary = session.summary()
    assert summary.winner == 1
    assert summary.forfeit_player == 0
    assert summary.faults == (3, 0)


def test_replay_round_trip_reproduces_hash(tmp_path: Path) -> None:
    path = tmp_path / "match.json"
    rules = RulesConfig(max_seconds=1.0)
    summary = run_headless("direct_ai", "bank_shot_ai", "classic", 7, rules, path)
    replayed = verify_replay(path)
    assert replayed["final_state_hash"] == summary.final_state_hash


def test_tournament_swaps_seats_and_writes_json_and_csv(tmp_path: Path) -> None:
    results = run_tournament(("direct_ai", "bank_shot_ai"), ("classic",), (0,))
    assert len(results) == 2
    assert {result.controllers for result in results} == {
        ("direct_ai", "bank_shot_ai"),
        ("bank_shot_ai", "direct_ai"),
    }
    output = tmp_path / "tournament.json"
    write_results(results, output)
    assert output.is_file()
    assert output.with_suffix(".csv").is_file()
    assert {row[0] for row in standings(results)} == {"direct_ai", "bank_shot_ai"}


def test_all_bundled_bots_run_on_all_maps_without_faults() -> None:
    rules = RulesConfig(max_seconds=0.5)
    for map_id in ("classic", "crossfire", "channels"):
        for first, second in (
            ("random_ai", "direct_ai"),
            ("template_ai", "bank_shot_ai"),
        ):
            summary = run_headless(first, second, map_id, rules=rules)
            assert summary.faults == (0, 0)


def test_generated_tournament_round_robin_on_shared_maps() -> None:
    import pytest

    from ricochet.runtime.cli import build_parser
    from ricochet.runtime.tournament import head_to_head, run_generated_tournament

    bots = ("template_ai", "random_ai", "direct_ai")
    rules = RulesConfig(max_seconds=1)
    results = run_generated_tournament(bots, rounds=2, seed=5, workers=1, rules=rules)
    assert len(results) == 3 * 2 * 2  # Pairings x rounds x seats.
    for first, second in (("template_ai", "random_ai"), ("random_ai", "direct_ai")):
        games = [r for r in results if set(r.controllers) == {first, second}]
        assert [r.map_id for r in games] == ["generated:5"] * 2 + ["generated:6"] * 2
        assert [r.controllers for r in games[:2]] == [(first, second), (second, first)]
    records = head_to_head(results)
    for (bot, opponent), (wins, draws, losses) in records.items():
        assert records[(opponent, bot)] == (losses, draws, wins)
        assert wins + draws + losses == 4
    assert sum(row[2] + row[3] + row[4] for row in standings(results)) == 2 * len(results)
    # Parallel workers play the same deterministic games, in the same order.
    parallel = run_generated_tournament(bots[:2], rounds=1, seed=5, workers=2, rules=rules)
    assert [(r.winner, r.final_state_hash) for r in parallel] == [
        (r.winner, r.final_state_hash) for r in results[:2]
    ]
    with pytest.raises(ValueError, match="unknown bots"):
        run_generated_tournament(("template_ai", "human"), rounds=1)
    assert build_parser().parse_args(["auto-tournament"]).rounds == 100
