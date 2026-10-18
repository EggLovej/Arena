from pathlib import Path

import pytest

from ricochet.game.config import RulesConfig
from ricochet.game.engine import GameEngine
from ricochet.game.maps import available_maps, load_map
from ricochet.game.types import ControlAction
from ricochet.runtime.loading import available_ais, load_controller
from ricochet.runtime.match import MatchSession, run_headless
from ricochet.runtime.replay import load_replay, replay_steps, verify_replay

ARENA_SEEDS = (3, 8, 12, 18, 21)  # The generated maps the Arena plays: GENERATED in arena_game.py.


def test_two_loads_of_a_bot_are_two_players() -> None:
    # A bot playing itself needs its own state in each seat.
    assert load_controller("direct_ai") is not load_controller("direct_ai")


def test_every_bot_plays_every_arena_map_without_faults() -> None:
    maps = (*available_maps(), *(f"generated:{seed}" for seed in ARENA_SEEDS))
    for map_id in maps:
        for bot in available_ais():
            summary = run_headless(bot, "direct_ai", map_id, rules=RulesConfig(max_seconds=1))
            assert summary.faults == (0, 0), (bot, map_id, summary.errors)


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


def test_dataclass_plugin_and_startup_failures(tmp_path: Path) -> None:
    (tmp_path / "typed.py").write_text(
        "from __future__ import annotations\nfrom dataclasses import dataclass\n"
        "from ricochet import ControlAction\n@dataclass\nclass Bot:\n    x: int = 0\n"
        "    def reset(self,context): pass\n    def act(self,obs): return ControlAction()\n"
        "def create_controller(): return Bot()\n"
    )
    assert load_controller("typed", tmp_path).x == 0
    one = run_headless("missing_ai", "direct_ai")
    assert one.winner == 1 and not one.invalid
    # When both fail the match is invalid, and its replay still plays back.
    path = tmp_path / "both.json"
    both = run_headless("missing_ai", "missing_ai", record_path=path)
    assert both.invalid
    assert verify_replay(path)["final_state_hash"] == both.final_state_hash


def test_replay_embeds_map_and_seeking_matches(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "replay.json"
    rules = RulesConfig(max_seconds=12)
    summary = run_headless(
        "reactive_defender_ai", "reactive_defender_ai", "classic", rules=rules, record_path=path
    )
    data = load_replay(path)
    assert "300" in data["snapshots"]
    # No installed map may be consulted during replay.
    import ricochet.game.maps

    monkeypatch.setattr(
        ricochet.game.maps,
        "load_map",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("map reload")),
    )
    final = list(replay_steps(data, start_tick=320))[-1][0]
    assert final.state_hash() == summary.final_state_hash
    assert verify_replay(path)["final_state_hash"] == summary.final_state_hash
    data["snapshots"]["300"]["tanks"][0]["health"] = 1
    with pytest.raises(ValueError, match="snapshot"):
        list(replay_steps(data, start_tick=320))
