"""Building blocks for controllers: their protocol, a base class and observation helpers.

Nothing here is required. Any object with ``reset(context)`` and ``act(observation)`` is a
controller (``Controller`` spells that out for type checkers); the base class and helpers
just keep bots from re-deriving the same game geometry.
"""

from __future__ import annotations

import heapq
import math
from typing import Protocol

from ricochet.game.config import RulesConfig
from ricochet.game.physics import segment_circle_time, segment_hits_wall, tank_position_is_clear
from ricochet.game.types import (
    ControlAction,
    GameObservation,
    MatchContext,
    ProjectileView,
    TankView,
    Vec2,
)


class Controller(Protocol):
    """What a match asks of a bot: ``reset`` once per match, then ``act`` every tick."""

    def reset(self, context: MatchContext) -> None: ...

    def act(self, observation: GameObservation) -> ControlAction: ...


class BaseController:
    """Stores the match context on ``reset``; subclasses implement ``act``.

    Override ``reset`` for extra per-match state and call ``super().reset(context)`` first.
    """

    context: MatchContext
    rules: RulesConfig
    player_id: int
    opponent_id: int

    def reset(self, context: MatchContext) -> None:
        self.context = context
        self.rules = context.rules
        self.player_id = context.player_id
        self.opponent_id = context.opponent_id

    def act(self, observation: GameObservation) -> ControlAction:
        raise NotImplementedError


def threats(observation: GameObservation) -> tuple[ProjectileView, ...]:
    """Projectiles that can damage this player: the opponent's, and its own after a bounce."""
    return tuple(
        projectile
        for projectile in observation.projectiles
        if projectile.owner_id != observation.player_id or projectile.bounced
    )


def on_collision_course(
    projectile: ProjectileView, target: Vec2, radius: float, seconds: float
) -> bool:
    """Whether the straight path over ``seconds`` passes within ``radius`` of ``target``."""
    end = projectile.position + projectile.velocity * seconds
    return segment_circle_time(projectile.position, end, target, radius) is not None


def line_of_sight(observation: GameObservation, start: Vec2, end: Vec2) -> bool:
    """Whether a zero-width line between the points misses every wall."""
    return not any(segment_hits_wall(start, end, wall) for wall in observation.walls)


def position_is_clear(observation: GameObservation, position: Vec2, radius: float) -> bool:
    """Whether a circle fits at ``position`` inside the arena without touching a wall."""
    return tank_position_is_clear(position, radius, observation)


def lead_target(shooter: Vec2, target: TankView, projectile_speed: float) -> Vec2:
    """Aim vector toward where ``target`` will be when a shot fired now arrives."""
    flight = (target.position - shooter).length() / projectile_speed
    return target.position + target.velocity * flight - shooter


def find_path(
    observation: GameObservation,
    start: Vec2,
    goal: Vec2,
    radius: float,
    spacing: float = 30,
    max_nodes: int = 1500,
) -> list[Vec2]:
    """A* over a grid of clear positions; the waypoints after ``start``.

    If the goal's own cell is blocked (a tank hugging a wall) or out of reach, the route ends
    at the explored cell closest to it, so callers always get somewhere useful to go.
    """

    def cell_of(point: Vec2) -> tuple[int, int]:
        return round(point.x / spacing), round(point.y / spacing)

    def point_of(cell: tuple[int, int]) -> Vec2:
        return Vec2(cell[0] * spacing, cell[1] * spacing)

    def remaining(cell: tuple[int, int]) -> int:
        return abs(cell[0] - target[0]) + abs(cell[1] - target[1])

    origin, target = cell_of(start), cell_of(goal)
    queue = [(0, origin)]
    costs, parents = {origin: 0}, {}
    found = origin
    while queue and len(costs) < max_nodes:
        _, cell = heapq.heappop(queue)
        if remaining(cell) < remaining(found):
            found = cell
        if cell == target:
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (cell[0] + dx, cell[1] + dy)
            if not position_is_clear(observation, point_of(neighbor), radius):
                continue
            cost = costs[cell] + 1
            if cost < costs.get(neighbor, math.inf):
                costs[neighbor], parents[neighbor] = cost, cell
                heapq.heappush(queue, (cost + remaining(neighbor), neighbor))
    route = []
    while found in parents:
        route.append(point_of(found))
        found = parents[found]
    return route[::-1]


__all__ = [
    "BaseController",
    "Controller",
    "find_path",
    "lead_target",
    "line_of_sight",
    "on_collision_course",
    "position_is_clear",
    "threats",
]
