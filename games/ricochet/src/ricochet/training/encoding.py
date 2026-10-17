from __future__ import annotations

import numpy as np

from ricochet.game.config import RulesConfig
from ricochet.game.types import ControlAction, GameObservation, TankView, Vec2

MAX_WALLS = 12
MAX_PROJECTILES = 24
ENCODER_VERSION = 2
ACTION_VERSION = 1
OBSERVATION_SIZE = 1 + 2 * 10 + MAX_WALLS * 5 + MAX_PROJECTILES * 8 + 8
ACTION_SIZE = 6


def check_capacity(rules: RulesConfig) -> None:
    """Encoder slots are fixed per encoder version; rules may not allow more objects."""
    if rules.max_walls > MAX_WALLS or rules.max_projectiles > MAX_PROJECTILES:
        raise ValueError("rules exceed encoder capacity")


def _scaled(value: float, maximum: float) -> float:
    return max(-1.0, min(1.0, value / maximum)) if maximum else 0.0


def _tank_features(tank: TankView, observation: GameObservation, rules: RulesConfig) -> list[float]:
    return [
        tank.position.x / observation.width * 2 - 1,
        tank.position.y / observation.height * 2 - 1,
        _scaled(tank.velocity.x, rules.tank_speed),
        _scaled(tank.velocity.y, rules.tank_speed),
        tank.aim.x,
        tank.aim.y,
        tank.health / rules.starting_health,
        tank.energy / rules.max_energy,
        min(1.0, tank.cooldown_remaining / rules.fire_cooldown_seconds),
        1.0 if tank.shield_active else 0.0,
    ]


def encode_observation(
    observation: GameObservation,
    rules: RulesConfig | None = None,
) -> np.ndarray:
    active_rules = rules or RulesConfig()
    check_capacity(active_rules)
    if len(observation.walls) > MAX_WALLS or len(observation.projectiles) > MAX_PROJECTILES:
        raise ValueError("observation exceeds encoder capacity")
    values: list[float] = [
        max(0.0, (observation.max_seconds - observation.time_seconds) / observation.max_seconds)
    ]
    values.extend(_tank_features(observation.me, observation, active_rules))
    values.extend(_tank_features(observation.opponent, observation, active_rules))
    for index in range(MAX_WALLS):
        if index < len(observation.walls):
            wall = observation.walls[index]
            values.extend(
                [
                    (wall.x + wall.width / 2) / observation.width * 2 - 1,
                    (wall.y + wall.height / 2) / observation.height * 2 - 1,
                    wall.width / observation.width,
                    wall.height / observation.height,
                    1.0,
                ]
            )
        else:
            values.extend((0.0,) * 5)
    ordered_projectiles = sorted(
        observation.projectiles,
        key=lambda projectile: (
            (projectile.position - observation.me.position).length(),
            projectile.id,
        ),
    )
    for index in range(MAX_PROJECTILES):
        if index < len(ordered_projectiles):
            projectile = ordered_projectiles[index]
            relative = projectile.position - observation.me.position
            values.extend(
                [
                    _scaled(relative.x, observation.width),
                    _scaled(relative.y, observation.height),
                    _scaled(projectile.velocity.x, active_rules.projectile_speed),
                    _scaled(projectile.velocity.y, active_rules.projectile_speed),
                    1.0 if projectile.owner_id == observation.player_id else -1.0,
                    1.0 if projectile.bounced else -1.0,
                    max(0.0, projectile.remaining_lifetime / active_rules.projectile_lifetime),
                    1.0,
                ]
            )
        else:
            values.extend((0.0,) * 8)
    relative = observation.opponent.position - observation.me.position
    velocity = observation.opponent.velocity - observation.me.velocity
    values.extend(
        [
            observation.me.shield_ticks_remaining / active_rules.shield_min_ticks,
            observation.opponent.shield_ticks_remaining / active_rules.shield_min_ticks,
            observation.width / (observation.width + 960),
            observation.height / (observation.height + 640),
            relative.x / observation.width,
            relative.y / observation.height,
            velocity.x / (2 * active_rules.tank_speed),
            velocity.y / (2 * active_rules.tank_speed),
        ]
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (OBSERVATION_SIZE,):
        raise AssertionError(f"encoder produced {result.shape}, expected {(OBSERVATION_SIZE,)}")
    return np.clip(result, -1.0, 1.0)


def decode_action(values: np.ndarray | list[float] | tuple[float, ...]) -> ControlAction:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (ACTION_SIZE,):
        raise ValueError(f"action must have shape {(ACTION_SIZE,)}")
    if not np.isfinite(vector).all():
        raise ValueError("action must contain finite values")
    vector = np.clip(vector, -1.0, 1.0)
    return ControlAction(
        move=Vec2(float(vector[0]), float(vector[1])).limited(),
        aim=Vec2(float(vector[2]), float(vector[3])).limited(),
        fire=bool(vector[4] > 0),
        shield=bool(vector[5] > 0),
    )
