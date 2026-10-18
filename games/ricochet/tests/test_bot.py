from dataclasses import replace

from ricochet.bot import find_path, line_of_sight, position_is_clear, threats
from ricochet.game.engine import GameEngine
from ricochet.game.maps import load_map
from ricochet.game.types import ProjectileView, Vec2


def _shot(identifier: int, owner: int, bounced: bool) -> ProjectileView:
    return ProjectileView(identifier, owner, Vec2(500, 100), Vec2(-420, 0), bounced, 1.0)


def test_threats_include_own_shot_only_after_it_bounces() -> None:
    observation = GameEngine(load_map("classic")).observation_for(0)
    shots = (_shot(1, 0, False), _shot(2, 0, True), _shot(3, 1, False))
    observation = replace(observation, projectiles=shots)
    assert [shot.id for shot in threats(observation)] == [2, 3]


def test_geometry_helpers_accept_observations() -> None:
    observation = GameEngine(load_map("classic")).observation_for(0)
    me, opponent = observation.me.position, observation.opponent.position
    inside_wall = Vec2(480, 320)  # Center of the classic block: never a clear cell.
    assert not line_of_sight(observation, me, opponent)  # The center block is in the way.
    assert line_of_sight(observation, me, Vec2(me.x, 40))
    assert not position_is_clear(observation, inside_wall, 18)
    route = find_path(observation, me, opponent, 20)
    assert route and (route[-1] - opponent).length() < 30
    assert all(position_is_clear(observation, point, 20) for point in route)
    # A goal no tank fits on: the route heads for the nearest cell it can reach.
    route = find_path(observation, me, inside_wall, 20)
    assert route and (route[-1] - inside_wall).length() < 140
