import math
from dataclasses import replace

import pytest

from ricochet import ControlAction, Vec2
from ricochet.game.config import RulesConfig
from ricochet.game.engine import GameEngine
from ricochet.game.maps import load_map
from ricochet.runtime.loading import load_controller


def test_intercept_meets_moving_target():
    bot = load_controller("sentinel_ai")
    intercept = bot.shot.__globals__["intercept"]
    for offset, velocity in [
        (Vec2(350, 100), Vec2(-90, 120)),
        (Vec2(150, 0), Vec2(180, 0)),
        (Vec2(0, 200), Vec2(0, 0)),
    ]:
        t = intercept(offset, velocity, 420)
        assert (offset + velocity * t).length() == pytest.approx(420 * t)


def test_shield_independent_deterministic_actions():
    engine = GameEngine(load_map("crossfire"))
    first, second = load_controller("sentinel_ai"), load_controller("sentinel_ai")
    first.reset(engine.context_for(0))
    second.reset(
        replace(engine.context_for(0), rules=replace(RulesConfig(), shield_activation_cost=1000))
    )
    for _ in range(25):
        observation = engine.observation_for(0)
        action = first.act(observation)
        assert action == second.act(observation)
        assert not action.shield
        assert all(
            math.isfinite(v) for v in (action.move.x, action.move.y, action.aim.x, action.aim.y)
        )
        engine.step({0: action, 1: ControlAction()})


def test_wall_trajectory_matches_live_bounce():
    engine = GameEngine(load_map("classic"))
    bot = load_controller("sentinel_ai")
    bot.reset(engine.context_for(0))
    trace = bot.shot.__globals__["trajectory"]
    # Fire toward the top boundary, away from both tanks and the center obstacle.
    engine.step({0: ControlAction(aim=Vec2(0, -1), fire=True)})
    shot = engine.observation_for(0).projectiles[0]
    segments = trace(
        shot.position,
        shot.velocity,
        shot.bounced,
        1.1,
        engine.rules.projectile_radius,
        engine.observation_for(0),
    )
    for _ in range(33):
        engine.step({})
    actual = engine.observation_for(0).projectiles[0]
    assert (actual.position - segments[-1][3]).length() < 1e-3
    assert actual.bounced


def test_projectile_search_chooses_escape_over_stationary_hit():
    from ricochet.game.types import ProjectileView

    engine = GameEngine(load_map("crossfire"))
    bot = load_controller("sentinel_ai")
    bot.reset(engine.context_for(0))
    obs = engine.observation_for(0)
    bullet = ProjectileView(99, 1, obs.me.position + Vec2(110, 0), Vec2(-420, 0), False, 2)
    obs = replace(obs, projectiles=(bullet,))
    action = bot.act(obs)
    assert abs(action.move.y) > 0.3
    assert not action.shield


def test_self_owned_returning_shot_is_a_threat():
    from ricochet.game.types import ProjectileView

    engine = GameEngine(load_map("crossfire"))
    bot = load_controller("sentinel_ai")
    bot.reset(engine.context_for(0))
    obs = engine.observation_for(0)
    shot = ProjectileView(100, 0, obs.me.position + Vec2(110, 0), Vec2(-420, 0), True, 2)
    assert abs(bot.act(replace(obs, projectiles=(shot,))).move.y) > 0.3
