from dataclasses import replace

from ricochet import BaseController
from ricochet.bot import find_path, line_of_sight, position_is_clear, threats
from ricochet.game.engine import GameEngine
from ricochet.game.maps import load_map
from ricochet.game.types import MapDefinition, ProjectileView, Vec2
from ricochet.runtime.loading import available_ais, load_controller


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
    assert not line_of_sight(observation, me, opponent)  # The center block is in the way.
    assert line_of_sight(observation, me, Vec2(me.x, 40))
    assert not position_is_clear(observation, Vec2(480, 320), 18)
    route = find_path(observation, me, opponent, 20)
    assert route and (route[-1] - opponent).length() < 30
    assert all(position_is_clear(observation, point, 20) for point in route)


def test_every_bundled_bot_uses_the_base_controller() -> None:
    for name in available_ais():
        assert isinstance(load_controller(name), BaseController), name


def test_map_serialization_round_trips_and_normalizes_numbers() -> None:
    definition = load_map("crossfire")
    assert MapDefinition.from_dict(definition.to_dict()) == definition
    rebuilt = MapDefinition.from_dict(
        definition.to_dict() | {"width": 960, "height": 640}  # Integers, as in map files.
    )
    assert rebuilt == definition and isinstance(rebuilt.width, float)


def test_engine_from_state_needs_no_map_file() -> None:
    engine = GameEngine(load_map("channels"))
    engine.step({})
    restored = GameEngine.from_state(engine.capture_state())
    assert restored.state_hash() == engine.state_hash()
    assert restored.map_definition == engine.map_definition


def test_find_path_heads_for_the_nearest_reachable_cell_when_goal_is_blocked() -> None:
    observation = GameEngine(load_map("classic")).observation_for(0)
    inside_wall = Vec2(480, 320)  # Center of the classic block: never a clear cell.
    route = find_path(observation, observation.me.position, inside_wall, 20)
    assert route and (route[-1] - inside_wall).length() < 140
