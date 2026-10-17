"""Seeded random arenas: a symmetric layout, then a slight disturbance.

The layout is mirror- or half-turn-symmetric, so neither seat starts with a structural
advantage. The disturbance then moves every free wall edge and both spawns by up to
``asymmetry`` pixels. Seat-swapped tournaments cancel whatever advantage that leaves.

A generated map is accepted only if it is playable: walls are thick enough, every gap
between walls (or between a wall and the border) is either closed or wide enough for a
tank, spawns have room, the tanks can reach each other and almost no open floor is sealed
off. ``generate_map(seed)`` is deterministic; ``load_map("generated:SEED")`` uses it.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

from ricochet.game.config import RulesConfig
from ricochet.game.maps import GENERATED_PREFIX, validate_map
from ricochet.game.types import MapDefinition, SpawnDefinition, Vec2, WallDefinition

Rect = tuple[float, float, float, float]  # left, top, right, bottom

_REACHABILITY_STEP = 8.0  # Grid spacing of the flood fill; below the narrowest corridor.


@dataclass(frozen=True)
class GeneratorConfig:
    width: float = 960.0
    height: float = 640.0
    min_wall_pairs: int = 3
    max_wall_pairs: int = 5
    center_piece_chance: float = 0.5
    asymmetry: float = 16.0  # Largest shift of any wall edge or spawn coordinate, in pixels.
    min_thickness: float = 24.0
    passage_clearance: float = 14.0  # Room beyond a tank diameter in every open gap.
    spawn_clearance: float = 40.0  # Free space between a spawned tank and the nearest wall.
    min_coverage: float = 0.04
    max_coverage: float = 0.22
    central_band: float = 0.5  # Middle share of the width that must contain some cover.
    min_reachable: float = 0.98  # Share of open floor connected to the spawns.


@dataclass(frozen=True)
class _Layout:
    walls: tuple[Rect, ...]
    spawns: tuple[Vec2, Vec2]


def generated_map_id(seed: int) -> str:
    return f"{GENERATED_PREFIX}{seed}"


@lru_cache(maxsize=256)
def generate_map(
    seed: int,
    config: GeneratorConfig | None = None,
    rules: RulesConfig | None = None,
) -> MapDefinition:
    config = config or GeneratorConfig()
    rules = rules or RulesConfig()
    rng = random.Random(seed)
    for _ in range(400):
        layout = _symmetric_layout(rng, config, rules)
        if layout is None or not _playable(layout, config, rules):
            continue
        for _ in range(12):
            candidate = _disturb(layout, rng, config)
            if _playable(candidate, config, rules):
                return _definition(seed, candidate, config)
    raise RuntimeError(f"no playable map found for seed {seed}")


def _symmetric_layout(rng: random.Random, config: GeneratorConfig, rules: RulesConfig):
    width, height = config.width, config.height
    gap = 2 * rules.tank_radius + config.passage_clearance
    mirror = rng.random() < 0.5  # Otherwise a half-turn about the arena center.

    def image(rect: Rect) -> Rect:
        left, top, right, bottom = rect
        if mirror:
            return width - right, top, width - left, bottom
        return width - right, height - bottom, width - left, height - top

    walls: list[Rect] = []
    if rng.random() < config.center_piece_chance:
        # Even sizes keep a centered piece on whole pixels, so it is its own image.
        w, h = (2 * round(side / 2) for side in _wall_size(rng, config))
        walls.append((width / 2 - w / 2, height / 2 - h / 2, width / 2 + w / 2, height / 2 + h / 2))
    for _ in range(rng.randint(config.min_wall_pairs, config.max_wall_pairs)):
        w, h = _wall_size(rng, config)
        # Mostly interior: border-hugging walls give little cover.
        x, y = rng.uniform(0.12 * width, width / 2), rng.uniform(0.08 * height, 0.92 * height)
        left = min(max(0.0, x - w / 2), width - w)
        top = min(max(0.0, y - h / 2), height - h)
        rect = _attach_to_border(
            tuple(round(v) for v in (left, top, left + w, top + h)),
            width,
            height,
            gap,
        )
        if mirror and image(rect)[0] - rect[2] < gap:
            # Too close to its own mirror image: merge both into one centered wall.
            walls.append((rect[0], rect[1], width - rect[0], rect[3]))
        else:
            walls.extend((rect, image(rect)))
    if len(walls) > rules.max_walls:
        return None
    radius = rules.tank_radius
    spawn = Vec2(round(rng.uniform(80, 180)), round(rng.uniform(radius + 60, height - radius - 60)))
    other = Vec2(width - spawn.x, spawn.y if mirror else height - spawn.y)
    return _Layout(tuple(walls), (spawn, other))


def _wall_size(rng: random.Random, config: GeneratorConfig) -> tuple[float, float]:
    if rng.random() < 0.65:
        thickness = rng.uniform(config.min_thickness, config.min_thickness + 16)
        length = rng.uniform(90, 260)
        return (length, thickness) if rng.random() < 0.5 else (thickness, length)
    return rng.uniform(50, 140), rng.uniform(50, 140)


def _attach_to_border(rect: Rect, width: float, height: float, reach: float) -> Rect:
    """Edges within ``reach`` of the border move onto it, closing a too-narrow slit."""
    left, top, right, bottom = rect
    return (
        0.0 if left < reach else left,
        0.0 if top < reach else top,
        width if width - right < reach else right,
        height if height - bottom < reach else bottom,
    )


def _disturb(layout: _Layout, rng: random.Random, config: GeneratorConfig) -> _Layout:
    """Shift every wall edge not on the border, and both spawns, independently."""
    width, height, amount = config.width, config.height, config.asymmetry

    def shift(value: float, fixed: bool) -> float:
        offset = rng.uniform(-amount, amount)
        return value if fixed else round(value + offset)

    walls = tuple(
        (
            shift(left, left <= 0),
            shift(top, top <= 0),
            shift(right, right >= width),
            shift(bottom, bottom >= height),
        )
        for left, top, right, bottom in layout.walls
    )
    spawns = tuple(Vec2(shift(spawn.x, False), shift(spawn.y, False)) for spawn in layout.spawns)
    return _Layout(walls, spawns)


def _distance(first: Rect, second: Rect) -> float:
    dx = max(0.0, second[0] - first[2], first[0] - second[2])
    dy = max(0.0, second[1] - first[3], first[1] - second[3])
    return math.hypot(dx, dy)


def _point_distance(point: Vec2, rect: Rect) -> float:
    dx = max(rect[0] - point.x, 0.0, point.x - rect[2])
    dy = max(rect[1] - point.y, 0.0, point.y - rect[3])
    return math.hypot(dx, dy)


def _playable(layout: _Layout, config: GeneratorConfig, rules: RulesConfig) -> bool:
    width, height = config.width, config.height
    gap = 2 * rules.tank_radius + config.passage_clearance
    walls = layout.walls
    if len(walls) > rules.max_walls:
        return False
    for left, top, right, bottom in walls:
        if min(right - left, bottom - top) < config.min_thickness:
            return False
        # Each edge either touches the border or leaves a passage.
        for space in (left, top, width - right, height - bottom):
            if space < 0 or 0 < space < gap:
                return False
    for index, first in enumerate(walls):
        for second in walls[index + 1 :]:
            if 0 < _distance(first, second) < gap:
                return False
    coverage = sum((right - left) * (bottom - top) for left, top, right, bottom in walls)
    if not config.min_coverage <= coverage / (width * height) <= config.max_coverage:
        return False
    band = ((1 - config.central_band) * width / 2, (1 + config.central_band) * width / 2)
    if not any(left < band[1] and right > band[0] for left, _, right, _ in walls):
        return False
    room = rules.tank_radius + config.spawn_clearance
    if any(_point_distance(spawn, wall) < room for spawn in layout.spawns for wall in walls):
        return False
    try:
        validate_map(_definition(0, layout, config), rules)
    except ValueError:
        return False
    return _connected(layout, config, rules)


def _connected(layout: _Layout, config: GeneratorConfig, rules: RulesConfig) -> bool:
    """Flood-fill tank-centre positions on a grid from one spawn."""
    radius, step = rules.tank_radius, _REACHABILITY_STEP
    columns = int((config.width - 2 * radius) // step) + 1
    rows = int((config.height - 2 * radius) // step) + 1
    blocked = bytearray(columns * rows)
    for left, top, right, bottom in layout.walls:
        # Only cells whose centre could lie within ``radius`` of this wall.
        first_column = max(0, math.ceil((left - 2 * radius) / step))
        last_column = min(columns - 1, math.floor(right / step))
        first_row = max(0, math.ceil((top - 2 * radius) / step))
        last_row = min(rows - 1, math.floor(bottom / step))
        for row in range(first_row, last_row + 1):
            y = radius + row * step
            dy = max(top - y, 0.0, y - bottom)
            for column in range(first_column, last_column + 1):
                x = radius + column * step
                dx = max(left - x, 0.0, x - right)
                if dx * dx + dy * dy < radius * radius:
                    blocked[row * columns + column] = 1

    def cell(point: Vec2) -> int:
        column = min(columns - 1, max(0, round((point.x - radius) / step)))
        row = min(rows - 1, max(0, round((point.y - radius) / step)))
        return row * columns + column

    start, goal = (cell(spawn) for spawn in layout.spawns)
    if blocked[start] or blocked[goal]:
        return False
    seen = bytearray(blocked)
    seen[start] = 1
    queue, reached = deque([start]), 1
    while queue:
        index = queue.popleft()
        column = index % columns
        for neighbor, inside in (
            (index - 1, column > 0),
            (index + 1, column < columns - 1),
            (index - columns, index >= columns),
            (index + columns, index < len(seen) - columns),
        ):
            if inside and not seen[neighbor]:
                seen[neighbor] = 1
                reached += 1
                queue.append(neighbor)
    open_cells = len(blocked) - blocked.count(1)
    return bool(seen[goal]) and reached >= config.min_reachable * open_cells


def _definition(seed: int, layout: _Layout, config: GeneratorConfig) -> MapDefinition:
    first, second = layout.spawns
    walls = sorted(layout.walls, key=lambda rect: (rect[0], rect[1]))
    return MapDefinition(
        id=generated_map_id(seed),
        name=f"Generated {seed}",
        width=config.width,
        height=config.height,
        spawns=(
            SpawnDefinition(0, first, (second - first).normalized(Vec2(1, 0))),
            SpawnDefinition(1, second, (first - second).normalized(Vec2(-1, 0))),
        ),
        walls=tuple(
            WallDefinition(f"wall_{index}", left, top, right - left, bottom - top)
            for index, (left, top, right, bottom) in enumerate(walls)
        ),
    )
