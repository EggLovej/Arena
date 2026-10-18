"""Maps: loading them from ``maps/`` (or generating them), saving, checking and spawn jitter."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from ricochet.game.config import RulesConfig
from ricochet.game.physics import circle_overlaps_wall
from ricochet.game.types import PLAYERS, MapDefinition, Vec2
from ricochet.resources import contributor_directory

GENERATED_PREFIX = "generated:"


def default_maps_directory() -> Path:
    return contributor_directory("maps")


def available_maps(directory: Path | None = None) -> tuple[str, ...]:
    root = directory or default_maps_directory()
    return tuple(sorted(path.stem for path in root.glob("*.json")))


def is_generated_map_id(map_id: str) -> bool:
    return map_id.startswith(GENERATED_PREFIX) and map_id[len(GENERATED_PREFIX) :].isdigit()


def load_map(
    map_id: str,
    directory: Path | None = None,
    rules: RulesConfig | None = None,
) -> MapDefinition:
    """A map file by its stem, or ``generated:SEED`` for a freshly generated arena."""
    if is_generated_map_id(map_id):
        from ricochet.game.mapgen import generate_map  # mapgen builds on this module.

        return generate_map(int(map_id[len(GENERATED_PREFIX) :]), rules=rules)
    root = directory or default_maps_directory()
    path = root / f"{map_id}.json"
    if not path.is_file():
        raise ValueError(f"unknown map '{map_id}'; available: {', '.join(available_maps(root))}")
    data = {"id": map_id, "name": map_id.replace("_", " ").title()} | json.loads(path.read_text())
    definition = MapDefinition.from_dict(data)
    # Authoring convenience: spawn aims in files need only point the right way.
    definition = replace(
        definition,
        spawns=tuple(replace(spawn, aim=spawn.aim.normalized()) for spawn in definition.spawns),
    )
    validate_map(definition, rules or RulesConfig())
    return definition


def save_map(definition: MapDefinition, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(definition.to_dict(), indent=2) + "\n")


class UniformRandom(Protocol):
    """The two sampling methods shared by ``random.Random`` and ``numpy.random.Generator``."""

    def random(self) -> float: ...

    def uniform(self, low: float, high: float) -> float: ...


def jitter_spawns(
    definition: MapDefinition,
    rules: RulesConfig,
    rng: UniformRandom,
    radius: float = 75.0,
    attempts: int = 32,
) -> MapDefinition:
    """Move each spawn within ``radius`` and randomize its aim; unchanged if no valid try."""
    for _ in range(attempts):
        spawns = []
        for spawn in definition.spawns:
            angle = rng.uniform(0, math.tau)
            distance = radius * math.sqrt(rng.random())
            aim_angle = rng.uniform(0, math.tau)
            spawns.append(
                replace(
                    spawn,
                    position=spawn.position
                    + Vec2(math.cos(angle) * distance, math.sin(angle) * distance),
                    aim=Vec2(math.cos(aim_angle), math.sin(aim_angle)),
                )
            )
        candidate = replace(definition, spawns=tuple(spawns))
        try:
            validate_map(candidate, rules)
            return candidate
        except ValueError:
            pass
    return definition


def validate_map(definition: MapDefinition, rules: RulesConfig) -> None:
    """Raise ValueError unless the map is well formed and both tanks can start on it."""
    values = [definition.width, definition.height]
    for wall in definition.walls:
        values.extend((wall.x, wall.y, wall.width, wall.height))
    for spawn in definition.spawns:
        values.extend((spawn.position.x, spawn.position.y, spawn.aim.x, spawn.aim.y))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("map coordinates must be finite")
    if definition.schema_version != 1:
        raise ValueError("only map schema_version 1 is supported")
    if definition.width <= 0 or definition.height <= 0:
        raise ValueError("map dimensions must be positive")
    player_ids = {spawn.player_id for spawn in definition.spawns}
    if len(definition.spawns) != 2 or player_ids != set(PLAYERS):
        raise ValueError("map must contain exactly one spawn for players 0 and 1")
    if len(definition.walls) > rules.max_walls:
        raise ValueError(f"map has more than {rules.max_walls} walls")
    wall_ids = [wall.id for wall in definition.walls]
    if len(wall_ids) != len(set(wall_ids)):
        raise ValueError("wall IDs must be unique")
    radius = rules.tank_radius
    for wall in definition.walls:
        if wall.width <= 0 or wall.height <= 0:
            raise ValueError(f"wall '{wall.id}' must have positive dimensions")
        outside = (
            wall.left < 0
            or wall.top < 0
            or wall.right > definition.width
            or wall.bottom > definition.height
        )
        if outside:
            raise ValueError(f"wall '{wall.id}' lies outside the arena")
    for spawn in definition.spawns:
        if spawn.aim.length() < 0.99:
            raise ValueError("spawn aim must be non-zero")
        if not radius <= spawn.position.x <= definition.width - radius:
            raise ValueError(f"player {spawn.player_id} spawn is outside the arena")
        if not radius <= spawn.position.y <= definition.height - radius:
            raise ValueError(f"player {spawn.player_id} spawn is outside the arena")
        if any(circle_overlaps_wall(spawn.position, radius, wall) for wall in definition.walls):
            raise ValueError(f"player {spawn.player_id} spawn overlaps a wall")
    distance = (definition.spawns[0].position - definition.spawns[1].position).length()
    if distance < 2 * radius:
        raise ValueError("player spawns overlap")
