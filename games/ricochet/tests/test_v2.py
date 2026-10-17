import json
import math
from dataclasses import replace

import numpy as np
import pytest

from ricochet.game.config import RulesConfig
from ricochet.game.engine import GameEngine, _ProjectileState
from ricochet.game.maps import load_map, validate_map
from ricochet.game.physics import rounded_wall_collision, segment_circle_time
from ricochet.game.types import ControlAction, Vec2, WallDefinition
from ricochet.runtime.loading import load_controller
from ricochet.runtime.match import run_headless
from ricochet.runtime.replay import load_replay, replay_steps, verify_replay
from ricochet.runtime.tournament import standings
from ricochet.training.encoding import OBSERVATION_SIZE, encode_observation
from ricochet.training.gym_env import RicochetEnv


def test_wall_touching_shot_bounces_instead_of_penetrating():
    engine = GameEngine(load_map("classic"))
    engine.tanks[0].position = Vec2(382, 320)
    events = []
    for tick in range(3):
        result = engine.step({0: ControlAction(aim=Vec2(1, 0), fire=tick == 0)})
        events.extend(result.events)
        assert all(p.position.x <= 396 + 1e-5 for p in engine.projectiles)
    assert any(e.kind == "bounced" for e in events)
    assert engine.tanks[0].health == 2


def test_boundary_shot_stays_inside_arena():
    engine = GameEngine(load_map("classic"))
    engine.tanks[0].position = Vec2(18, 100)
    for tick in range(3):
        engine.step({0: ControlAction(aim=Vec2(-1, 0), fire=tick == 0)})
        assert all(p.position.x >= 4 - 1e-5 for p in engine.projectiles)


def test_inside_circle_is_immediate_contact():
    assert segment_circle_time(Vec2(0, 0), Vec2(1, 0), Vec2(0, 0), 22) == 0


def test_rounded_corner_and_tangent_contacts():
    wall = WallDefinition("test", 100, 100, 100, 100)
    contact = rounded_wall_collision(Vec2(80, 80), Vec2(110, 110), 10, wall)
    assert contact is not None
    assert contact.normal.x == pytest.approx(-math.sqrt(0.5))
    assert contact.normal.y == pytest.approx(-math.sqrt(0.5))
    assert rounded_wall_collision(Vec2(80, 90), Vec2(220, 90), 10, wall) is None
    contact = rounded_wall_collision(Vec2(110, 110), Vec2(120, 110), 10, wall)
    assert contact.t == 0


def test_tanks_separate_at_boundary_and_order_independent():
    first = GameEngine(load_map("crossfire"))
    first.tanks[0].position = Vec2(18, 100)
    first.tanks[1].position = Vec2(54, 100)
    second = first.clone()
    actions = {0: ControlAction(), 1: ControlAction(move=Vec2(-1, 0))}
    first.step(actions)
    second.step(dict(reversed(list(actions.items()))))
    assert (first.tanks[1].position - first.tanks[0].position).length() >= 36 - 1e-7
    assert first.state_hash() == second.state_hash()


def test_shield_prepayment_commitment_release_and_reactivation():
    engine = GameEngine(load_map("classic"))
    result = engine.step({0: ControlAction(shield=True, fire=True)})
    assert engine.tanks[0].energy == pytest.approx(76.8)
    assert engine.tanks[0].shield_ticks_remaining == 5
    assert [e.kind for e in result.events] == ["shield_activated"]
    for _ in range(5):
        engine.step({})
        assert engine.tanks[0].shield_active
    result = engine.step({})
    assert not engine.tanks[0].shield_active
    assert any(e.kind == "shield_deactivated" for e in result.events)
    before = engine.tanks[0].energy
    engine.step({0: ControlAction(shield=True)})
    assert engine.tanks[0].energy == pytest.approx(before + 0.4 - 23.2)


def test_shield_hold_and_insufficient_start_energy():
    engine = GameEngine(load_map("classic"))
    engine.tanks[0].energy = 22
    engine.step({0: ControlAction(shield=True)})
    assert not engine.tanks[0].shield_active
    engine.tanks[0].energy = 100
    for _ in range(6):
        engine.step({0: ControlAction(shield=True)})
    before = engine.tanks[0].energy
    engine.step({0: ControlAction(shield=True)})
    assert engine.tanks[0].energy == pytest.approx(before + 0.4 - 1.2)


def test_capture_restore_json_and_clone_do_not_alias():
    engine = GameEngine(load_map("crossfire"))
    engine.step({0: ControlAction(fire=True)})
    clone = engine.clone()
    restored = GameEngine(load_map("classic"))
    restored.restore_state(json.loads(json.dumps(engine.capture_state())))
    for _ in range(50):
        actions = {0: ControlAction(move=Vec2(0.2, 0), fire=True), 1: ControlAction(shield=True)}
        engine.step(actions)
        clone.step(actions)
        restored.step(actions)
        assert engine.state_hash() == clone.state_hash() == restored.state_hash()
    clone.tanks[0].health = 0
    assert engine.tanks[0].health != 0
    assert (
        GameEngine(load_map("classic")).state_hash()
        != GameEngine(load_map("crossfire")).state_hash()
    )


def test_self_hit_reward_is_negative_and_external_cutoff_truncates():
    env = RicochetEnv(opponent="template_ai", seat_mode=0)
    env.reset(seed=0)
    p = env.engine.tanks[0].position
    env.engine.projectiles = [_ProjectileState(99, 0, Vec2(p.x - 30, p.y), Vec2(420, 0), True, 5)]
    _, reward, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
    assert reward == pytest.approx(-0.1)
    env = RicochetEnv(max_episode_steps=1)
    env.reset(seed=0)
    _, _, terminated, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
    assert truncated and not terminated


def test_seed_sequence_advances_and_reproduces():
    env = RicochetEnv(opponent="random_ai", maps=("classic", "crossfire"), jitter_probability=0.5)

    def sequence():
        return [env.reset(seed=42 if i == 0 else None)[1]["episode_seed"] for i in range(5)]

    first = sequence()
    assert len(set(first)) == 5
    assert first == sequence()


def test_encoder_versions_and_capacity():
    engine = GameEngine(load_map("classic"))
    vector = encode_observation(engine.observation_for(0))
    assert vector.shape == (281,) == (OBSERVATION_SIZE,)
    assert vector[-6] == pytest.approx(0.5) and vector[-5] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="capacity"):
        RicochetEnv(rules=RulesConfig(max_projectiles=25))


def test_dataclass_plugin_and_startup_failures(tmp_path):
    (tmp_path / "typed.py").write_text(
        "from __future__ import annotations\nfrom dataclasses import dataclass\n"
        "from ricochet import ControlAction\n@dataclass\nclass Bot:\n    x: int = 0\n"
        "    def reset(self,context): pass\n    def act(self,obs): return ControlAction()\n"
        "def create_controller(): return Bot()\n"
    )
    assert load_controller("typed", tmp_path).x == 0
    one = run_headless("missing_ai", "direct_ai")
    assert one.winner == 1 and not one.invalid
    both = run_headless("missing_ai", "missing_ai")
    assert both.invalid and standings([both]) == []


def test_replay_embeds_map_and_seeking_matches(tmp_path, monkeypatch):
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


def test_invalid_replay_and_v1_rejection(tmp_path):
    path = tmp_path / "bad.json"
    summary = run_headless("missing_ai", "missing_ai", record_path=path)
    assert verify_replay(path)["final_state_hash"] == summary.final_state_hash
    path.write_text('{"schema_version":1}')
    with pytest.raises(ValueError, match="afc5ee0"):
        load_replay(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0])
def test_invalid_rules(value):
    with pytest.raises(ValueError):
        RulesConfig(tank_speed=value)


def test_nonfinite_map_rejected():
    definition = load_map("classic")
    with pytest.raises(ValueError, match="finite"):
        validate_map(replace(definition, width=float("nan")), RulesConfig())


def test_parallel_contract():
    from pettingzoo.test import parallel_api_test

    from ricochet.training.parallel_env import RicochetParallelEnv

    parallel_api_test(RicochetParallelEnv(rules=RulesConfig(max_seconds=0.1)), num_cycles=20)
