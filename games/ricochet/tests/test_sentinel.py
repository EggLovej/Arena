from dataclasses import replace

import pytest

from ricochet import ControlAction, Vec2
from ricochet.game.engine import GameEngine
from ricochet.game.maps import load_map
from ricochet.game.types import ProjectileView
from ricochet.runtime.loading import load_controller


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


@pytest.mark.parametrize(
    ("owner", "bounced"), [(1, False), (0, True)], ids=["opponent's shot", "own bounced shot"]
)
def test_sentinel_dodges_a_shot_that_can_hit_it(owner: int, bounced: bool) -> None:
    engine = GameEngine(load_map("crossfire"))
    bot = load_controller("sentinel_ai")
    bot.reset(engine.context_for(0))
    obs = engine.observation_for(0)
    shot = ProjectileView(99, owner, obs.me.position + Vec2(110, 0), Vec2(-420, 0), bounced, 2)
    assert abs(bot.act(replace(obs, projectiles=(shot,))).move.y) > 0.3
