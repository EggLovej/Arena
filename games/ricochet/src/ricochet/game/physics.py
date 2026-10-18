"""Collision geometry: tanks are circles, shots are small circles swept along their path.

A swept collision's time ``t`` is a fraction of the move: 0 at its start, 1 at its end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from ricochet.game.types import Vec2, WallDefinition

EPSILON = 1e-7


class Arena(Protocol):
    """Bounds and walls: satisfied by both ``MapDefinition`` and ``GameObservation``."""

    @property
    def width(self) -> float: ...

    @property
    def height(self) -> float: ...

    @property
    def walls(self) -> tuple[WallDefinition, ...]: ...


@dataclass(frozen=True)
class Collision:
    """Where a move first touches a surface: at fraction ``t`` of it, against ``normal``."""

    t: float
    normal: Vec2


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def circle_overlaps_wall(center: Vec2, radius: float, wall: WallDefinition) -> bool:
    closest_x = clamp(center.x, wall.left, wall.right)
    closest_y = clamp(center.y, wall.top, wall.bottom)
    return (center.x - closest_x) ** 2 + (center.y - closest_y) ** 2 < radius**2 - EPSILON


def tank_position_is_clear(position: Vec2, radius: float, arena: Arena) -> bool:
    """Whether a circle fits at ``position``, inside the border and clear of every wall."""
    if not radius <= position.x <= arena.width - radius:
        return False
    if not radius <= position.y <= arena.height - radius:
        return False
    return not any(circle_overlaps_wall(position, radius, wall) for wall in arena.walls)


def slide_circle(start: Vec2, displacement: Vec2, radius: float, arena: Arena) -> Vec2:
    """Move along x, then y, skipping an axis that would end in a wall: tanks slide along walls."""
    position = start
    x_candidate = Vec2(clamp(start.x + displacement.x, radius, arena.width - radius), start.y)
    if tank_position_is_clear(x_candidate, radius, arena):
        position = x_candidate
    y_candidate = Vec2(
        position.x,
        clamp(start.y + displacement.y, radius, arena.height - radius),
    )
    if tank_position_is_clear(y_candidate, radius, arena):
        position = y_candidate
    return position


def resolve_tank_overlap(
    first: Vec2,
    second: Vec2,
    radius: float,
    arena: Arena,
) -> tuple[Vec2, Vec2]:
    """Push two overlapping tanks apart, both halfway or one all the way if the other is stuck.

    Returns the positions unchanged if neither can move.
    """
    delta = second - first
    distance = delta.length()
    minimum = radius * 2
    if distance >= minimum - EPSILON:
        return first, second
    direction = delta.normalized(Vec2(1.0, 0.0))
    correction = direction * ((minimum - distance) / 2 + EPSILON)
    proposed_first = first - correction
    proposed_second = second + correction
    first_clear = tank_position_is_clear(proposed_first, radius, arena)
    second_clear = tank_position_is_clear(proposed_second, radius, arena)
    if first_clear and second_clear:
        return proposed_first, proposed_second
    if first_clear:
        candidate = first - correction * 2
        if tank_position_is_clear(candidate, radius, arena):
            return candidate, second
    if second_clear:
        candidate = second + correction * 2
        if tank_position_is_clear(candidate, radius, arena):
            return first, candidate
    return first, second


def segment_hits_wall(start: Vec2, end: Vec2, wall: WallDefinition) -> bool:
    """Whether the zero-width segment touches the wall rectangle (slab test)."""
    direction = end - start
    entry, exit_time = 0.0, 1.0
    for origin, delta, low, high in (
        (start.x, direction.x, wall.left, wall.right),
        (start.y, direction.y, wall.top, wall.bottom),
    ):
        if abs(delta) < 1e-9:
            if origin < low or origin > high:
                return False
            continue
        first, second = (low - origin) / delta, (high - origin) / delta
        if first > second:
            first, second = second, first
        entry, exit_time = max(entry, first), min(exit_time, second)
        if entry > exit_time:
            return False
    return 0 <= entry <= 1


def earliest_wall_collision(
    start: Vec2,
    end: Vec2,
    radius: float,
    arena: Arena,
) -> Collision | None:
    """The first contact, if any, of a circle of ``radius`` moving from ``start`` to ``end``
    with the border or a wall."""
    candidates: list[Collision] = []
    direction = end - start
    if direction.x < -EPSILON and end.x < radius:
        candidates.append(Collision((radius - start.x) / direction.x, Vec2(1, 0)))
    if direction.x > EPSILON and end.x > arena.width - radius:
        candidates.append(Collision((arena.width - radius - start.x) / direction.x, Vec2(-1, 0)))
    if direction.y < -EPSILON and end.y < radius:
        candidates.append(Collision((radius - start.y) / direction.y, Vec2(0, 1)))
    if direction.y > EPSILON and end.y > arena.height - radius:
        candidates.append(Collision((arena.height - radius - start.y) / direction.y, Vec2(0, -1)))
    for wall in arena.walls:
        collision = rounded_wall_collision(start, end, radius, wall)
        if collision is not None:
            candidates.append(collision)
    valid = [candidate for candidate in candidates if -EPSILON <= candidate.t <= 1 + EPSILON]
    return min(valid, key=lambda item: item.t) if valid else None


def segment_circle_time(start: Vec2, end: Vec2, center: Vec2, radius: float) -> float | None:
    """When a point moving from ``start`` to ``end`` comes within ``radius`` of ``center``.

    A fraction of the move from 0 to 1; 0 if it starts within, None if it never gets there.
    """
    direction = end - start
    relative = start - center
    a = direction.dot(direction)
    if a <= EPSILON:
        return 0.0 if relative.dot(relative) <= radius**2 else None
    b = 2 * relative.dot(direction)
    c = relative.dot(relative) - radius**2
    if c <= EPSILON:
        return 0.0
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None
    root = math.sqrt(discriminant)
    times = [time for time in ((-b - root) / (2 * a), (-b + root) / (2 * a)) if 0 <= time <= 1]
    return min(times) if times else None


def reflect(vector: Vec2, normal: Vec2) -> Vec2:
    """``vector`` bounced off a surface with unit ``normal``."""
    return vector - normal * (2 * vector.dot(normal))


def rounded_wall_collision(
    start: Vec2, end: Vec2, radius: float, wall: WallDefinition
) -> Collision | None:
    """Sweep against flat faces and quarter-circle corners, including penetration."""
    delta = end - start
    closest = Vec2(clamp(start.x, wall.left, wall.right), clamp(start.y, wall.top, wall.bottom))
    offset = start - closest
    if offset.length() < radius - EPSILON:
        if offset.length() > EPSILON:
            return Collision(0, offset.normalized())
        faces = [
            (abs(start.x - wall.left), Vec2(-1, 0)),
            (abs(start.x - wall.right), Vec2(1, 0)),
            (abs(start.y - wall.top), Vec2(0, -1)),
            (abs(start.y - wall.bottom), Vec2(0, 1)),
        ]
        return Collision(0, min(faces, key=lambda item: item[0])[1])
    contacts = []
    for axis, coordinate, low, high, normal in (
        (0, wall.left - radius, wall.top, wall.bottom, Vec2(-1, 0)),
        (0, wall.right + radius, wall.top, wall.bottom, Vec2(1, 0)),
        (1, wall.top - radius, wall.left, wall.right, Vec2(0, -1)),
        (1, wall.bottom + radius, wall.left, wall.right, Vec2(0, 1)),
    ):
        speed = delta.x if axis == 0 else delta.y
        origin = start.x if axis == 0 else start.y
        if abs(speed) <= EPSILON or delta.dot(normal) >= -EPSILON:
            continue
        t = (coordinate - origin) / speed
        point = start + delta * t
        other = point.y if axis == 0 else point.x
        if 0 <= t <= 1 and low - EPSILON <= other <= high + EPSILON:
            contacts.append(Collision(t, normal))
    for x, y, sx, sy in (
        (wall.left, wall.top, -1, -1),
        (wall.right, wall.top, 1, -1),
        (wall.left, wall.bottom, -1, 1),
        (wall.right, wall.bottom, 1, 1),
    ):
        center = Vec2(x, y)
        t = segment_circle_time(start, end, center, radius)
        if t is None:
            continue
        offset = start + delta * t - center
        normal = offset.normalized()
        if offset.x * sx >= -EPSILON and offset.y * sy >= -EPSILON and delta.dot(normal) < -EPSILON:
            contacts.append(Collision(t, normal))
    return min(contacts, key=lambda c: c.t) if contacts else None
