from dataclasses import FrozenInstanceError

import pytest

from ricochet.game.config import RulesConfig
from ricochet.game.engine import GameEngine, _ProjectileState
from ricochet.game.maps import load_map
from ricochet.game.types import ControlAction, Vec2


def test_action_dictionary_order_does_not_change_state() -> None:
    first = GameEngine(load_map("crossfire"))
    second = GameEngine(load_map("crossfire"))
    actions = {
        0: ControlAction(Vec2(1, 0.3), Vec2(1, 0), True),
        1: ControlAction(Vec2(-1, -0.3), Vec2(-1, 0), True),
    }
    first.step(actions)
    second.step({1: actions[1], 0: actions[0]})
    assert first.state_hash() == second.state_hash()


def test_observation_is_immutable() -> None:
    observation = GameEngine(load_map("classic")).observation_for(0)
    with pytest.raises(FrozenInstanceError):
        observation.tanks[0].health = 99  # type: ignore[misc]


def test_energy_is_shared_by_movement_and_firing() -> None:
    engine = GameEngine(load_map("crossfire"))
    engine.step({0: ControlAction(Vec2(1, 0), Vec2(1, 0), True)})
    expected = 100 - engine.rules.movement_energy_per_second * engine.rules.dt - 32
    assert engine.tanks[0].energy == pytest.approx(expected)


def test_firing_never_leaves_energy_below_zero() -> None:
    engine = GameEngine(load_map("classic"))
    rules = engine.rules
    # Regeneration adds up in floats, so a tank can be a hair short of the cost and still fire.
    engine.tanks[0].energy = rules.fire_energy - rules.energy_regen_per_second * rules.dt - 1e-12
    engine.step({0: ControlAction(fire=True)})
    assert engine.projectiles and engine.tanks[0].energy == 0


def test_shield_blocks_projectile_damage() -> None:
    engine = GameEngine(load_map("crossfire"))
    engine.tanks[0].position = Vec2(380, 320)
    engine.tanks[1].position = Vec2(580, 320)
    engine.tanks[0].aim = Vec2(1, 0)
    blocked = False
    for tick in range(30):
        result = engine.step(
            {
                0: ControlAction(aim=Vec2(1, 0), fire=tick == 0),
                1: ControlAction(aim=Vec2(-1, 0), shield=True),
            }
        )
        blocked |= any(event.kind == "blocked" for event in result.events)
        if blocked:
            break
    assert blocked
    assert engine.tanks[1].health == 3


def test_simultaneous_fatal_hits_are_a_draw() -> None:
    engine = GameEngine(load_map("crossfire"))
    engine.tanks[0].position = Vec2(380, 320)
    engine.tanks[1].position = Vec2(580, 320)
    engine.tanks[0].health = engine.tanks[1].health = 1
    result = None
    for tick in range(30):
        result = engine.step(
            {
                0: ControlAction(aim=Vec2(1, 0), fire=tick == 0),
                1: ControlAction(aim=Vec2(-1, 0), fire=tick == 0),
            }
        )
        if result.terminated:
            break
    assert result is not None and result.terminated
    assert result.winner is None
    assert engine.tanks[0].health == engine.tanks[1].health == 0


def test_projectile_can_hit_owner_after_bounce() -> None:
    engine = GameEngine(load_map("crossfire"))
    engine.tanks[0].position = Vec2(100, 320)
    engine.tanks[0].aim = Vec2(-1, 0)
    hit_owner = False
    for tick in range(60):
        result = engine.step({0: ControlAction(aim=Vec2(-1, 0), fire=tick == 0)})
        hit_owner |= any(event.kind == "hit" and event.player_id == 0 for event in result.events)
        if hit_owner:
            break
    assert hit_owner
    assert engine.tanks[0].health == 2


def test_second_wall_contact_removes_bounced_projectile() -> None:
    engine = GameEngine(load_map("classic"))
    engine.projectiles.append(_ProjectileState(99, 0, Vec2(480, 100), Vec2(0, -420), True, 5))
    removed = False
    for _ in range(20):
        result = engine.step({})
        removed |= any(event.kind == "removed_after_bounce" for event in result.events)
        if removed:
            break
    assert removed
    assert not engine.projectiles


def test_swept_projectile_hits_at_high_speed() -> None:
    rules = RulesConfig(projectile_speed=6000)
    engine = GameEngine(load_map("crossfire", rules=rules), rules)
    engine.tanks[0].position = Vec2(380, 320)
    engine.tanks[1].position = Vec2(550, 320)
    engine.step({0: ControlAction(aim=Vec2(1, 0), fire=True)})
    assert engine.tanks[1].health == 2


def test_tank_slides_along_wall() -> None:
    engine = GameEngine(load_map("classic"))
    engine.tanks[0].position = Vec2(381, 300)
    before = engine.tanks[0].position
    engine.step({0: ControlAction(move=Vec2(1, 1), aim=Vec2(1, 0))})
    after = engine.tanks[0].position
    assert after.x == pytest.approx(before.x)
    assert after.y > before.y


def test_time_limit_is_a_terminal_draw() -> None:
    rules = RulesConfig(max_seconds=1 / 30)
    engine = GameEngine(load_map("classic", rules=rules), rules)
    result = engine.step({})
    assert result.timed_out
    assert result.terminated
    assert result.winner is None
