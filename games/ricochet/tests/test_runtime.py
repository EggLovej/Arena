from pathlib import Path

from ricochet.game.config import RulesConfig
from ricochet.game.engine import GameEngine
from ricochet.game.maps import load_map
from ricochet.game.types import ControlAction
from ricochet.runtime.loading import available_ais, load_controller
from ricochet.runtime.match import MatchSession, run_headless
from ricochet.runtime.replay import verify_replay


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


def test_all_bundled_bots_run_on_all_maps_without_faults() -> None:
    rules = RulesConfig(max_seconds=0.5)
    for map_id in ("classic", "crossfire", "channels"):
        for first, second in (
            ("random_ai", "direct_ai"),
            ("template_ai", "bank_shot_ai"),
        ):
            summary = run_headless(first, second, map_id, rules=rules)
            assert summary.faults == (0, 0)
