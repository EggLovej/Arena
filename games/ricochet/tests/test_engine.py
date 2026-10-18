import json
import math

import pytest

from ricochet.game.config import RulesConfig
from ricochet.game.engine import GameEngine, _ProjectileState
from ricochet.game.maps import load_map
from ricochet.game.physics import rounded_wall_collision
from ricochet.game.types import ControlAction, Vec2, WallDefinition


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


def test_the_shield_costs_what_the_rules_say() -> None:
    engine = GameEngine(load_map("classic"))
    tank = engine.tanks[0]
    # Raising it costs 23.2 at once: 16, plus six prepaid ticks at 1.2. It can't fire meanwhile.
    result = engine.step({0: ControlAction(shield=True, fire=True)})
    assert [event.kind for event in result.events] == ["shield_activated"]
    assert tank.energy == pytest.approx(100 - 23.2)
    for _ in range(5):  # The prepaid ticks keep it up after it is let go.
        engine.step({})
        assert tank.shield_active
    engine.step({})
    assert not tank.shield_active
    before = tank.energy
    engine.step({0: ControlAction(shield=True)})
    assert tank.energy == pytest.approx(before + 0.4 - 23.2)  # Raising it again pays again.
    for _ in range(5):
        engine.step({0: ControlAction(shield=True)})
    before = tank.energy
    engine.step({0: ControlAction(shield=True)})
    assert tank.energy == pytest.approx(before + 0.4 - 1.2)  # Held past its six ticks.
    engine.step({})
    tank.energy = 22  # 22.4 after this tick's regeneration: not enough.
    engine.step({0: ControlAction(shield=True)})
    assert not tank.shield_active


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


def test_own_shot_bounces_off_a_wall_and_can_hit_its_owner() -> None:
    engine = GameEngine(load_map("classic"))
    engine.tanks[0].position = Vec2(382, 320)  # Just left of the centre block.
    events = []
    for tick in range(3):
        result = engine.step({0: ControlAction(aim=Vec2(1, 0), fire=tick == 0)})
        events.extend(result.events)
        assert all(p.position.x <= 396 + 1e-5 for p in engine.projectiles)  # Never inside.
    assert any(e.kind == "bounced" for e in events)
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


def test_boundary_shot_stays_inside_arena() -> None:
    engine = GameEngine(load_map("classic"))
    engine.tanks[0].position = Vec2(18, 100)
    for tick in range(3):
        engine.step({0: ControlAction(aim=Vec2(-1, 0), fire=tick == 0)})
        assert all(p.position.x >= 4 - 1e-5 for p in engine.projectiles)


def test_swept_projectile_hits_at_high_speed() -> None:
    rules = RulesConfig(projectile_speed=6000)
    engine = GameEngine(load_map("crossfire", rules=rules), rules)
    engine.tanks[0].position = Vec2(380, 320)
    engine.tanks[1].position = Vec2(550, 320)
    engine.step({0: ControlAction(aim=Vec2(1, 0), fire=True)})
    assert engine.tanks[1].health == 2


def test_rounded_corner_and_tangent_contacts() -> None:
    wall = WallDefinition("test", 100, 100, 100, 100)
    contact = rounded_wall_collision(Vec2(80, 80), Vec2(110, 110), 10, wall)
    assert contact is not None
    assert contact.normal.x == pytest.approx(-math.sqrt(0.5))
    assert contact.normal.y == pytest.approx(-math.sqrt(0.5))
    assert rounded_wall_collision(Vec2(80, 90), Vec2(220, 90), 10, wall) is None
    contact = rounded_wall_collision(Vec2(110, 110), Vec2(120, 110), 10, wall)
    assert contact.t == 0


def test_tank_slides_along_wall() -> None:
    engine = GameEngine(load_map("classic"))
    engine.tanks[0].position = Vec2(381, 300)
    before = engine.tanks[0].position
    engine.step({0: ControlAction(move=Vec2(1, 1), aim=Vec2(1, 0))})
    after = engine.tanks[0].position
    assert after.x == pytest.approx(before.x)
    assert after.y > before.y


def test_tanks_separate_at_boundary_and_order_independent() -> None:
    first = GameEngine(load_map("crossfire"))
    first.tanks[0].position = Vec2(18, 100)
    first.tanks[1].position = Vec2(54, 100)
    second = GameEngine.from_state(first.capture_state())
    actions = {0: ControlAction(), 1: ControlAction(move=Vec2(-1, 0))}
    first.step(actions)
    second.step(dict(reversed(list(actions.items()))))
    assert (first.tanks[1].position - first.tanks[0].position).length() >= 36 - 1e-7
    assert first.state_hash() == second.state_hash()


def test_time_limit_is_a_terminal_draw() -> None:
    rules = RulesConfig(max_seconds=1 / 30)
    engine = GameEngine(load_map("classic", rules=rules), rules)
    result = engine.step({})
    assert result.timed_out
    assert result.terminated
    assert result.winner is None


def test_captured_state_restores_exactly_without_the_map_file() -> None:
    engine = GameEngine(load_map("crossfire"))
    engine.step({0: ControlAction(fire=True)})
    copy = GameEngine.from_state(engine.capture_state())
    restored = GameEngine(load_map("classic"))
    restored.restore_state(json.loads(json.dumps(engine.capture_state())))
    assert copy.map_definition == restored.map_definition == engine.map_definition
    for _ in range(50):
        actions = {0: ControlAction(move=Vec2(0.2, 0), fire=True), 1: ControlAction(shield=True)}
        engine.step(actions)
        copy.step(actions)
        restored.step(actions)
        assert engine.state_hash() == copy.state_hash() == restored.state_hash()
    copy.tanks[0].health = 0
    assert engine.tanks[0].health != 0
    assert (
        GameEngine(load_map("classic")).state_hash()
        != GameEngine(load_map("crossfire")).state_hash()
    )
