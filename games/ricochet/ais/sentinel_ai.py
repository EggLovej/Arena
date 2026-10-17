"""Deterministic analytical controller: navigation, interception and receding-horizon evasion.

No learned parameters, hidden engine state, opponent-name lookup, or shield dependency.
"""

from __future__ import annotations

import heapq
import math

from ricochet import BaseController, ControlAction, Vec2
from ricochet.game.physics import (
    earliest_wall_collision,
    reflect,
    segment_circle_time,
    slide_circle,
    tank_position_is_clear,
)

ZERO = Vec2(0, 0)


def intercept(offset, velocity, speed):
    """Smallest positive solution of |offset + velocity*t| = speed*t."""
    a = velocity.dot(velocity) - speed * speed
    b, c = 2 * offset.dot(velocity), offset.dot(offset)
    if abs(a) < 1e-9:
        return -c / b if b < -1e-9 else math.sqrt(c) / speed
    disc = b * b - 4 * a * c
    roots = [] if disc < 0 else [(-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)]
    return min((t for t in roots if t > 0), default=math.sqrt(c) / speed)


def trajectory(position, velocity, bounced, seconds, radius, arena):
    """Exact wall-only swept-circle flight, split at its one permitted reflection."""
    result, elapsed = [], 0.0
    for _ in range(2):
        end = position + velocity * seconds
        hit = earliest_wall_collision(position, end, radius, arena)
        duration = seconds * max(0, hit.t) if hit else seconds
        stop = position + velocity * duration
        result.append((elapsed, elapsed + duration, position, stop, bounced))
        if hit is None or bounced:
            break
        elapsed += duration
        seconds -= duration
        position = stop + hit.normal * 1e-5
        velocity = reflect(velocity, hit.normal)
        bounced = True
    return result


class Sentinel(BaseController):
    def reset(self, context):
        super().reset(context)
        self.nodes = None
        self.graph = []
        self.waypoint = None
        self.previous_move = ZERO
        self.shots_fired = 0
        self.use_shield = False
        self.use_banks = True
        self.enemy_shields = False
        self.enemy_attacks = False
        self.short_range_shooter = False

    def clear(self, obs, start, end, radius):
        return earliest_wall_collision(start, end, radius, obs) is None

    def navigation(self, obs):
        radius = self.rules.tank_radius + 1
        if self.nodes is None:
            self.nodes = []
            for wall in obs.walls:
                for x in (wall.left - radius - 1, wall.right + radius + 1):
                    for y in (wall.top - radius - 1, wall.bottom + radius + 1):
                        point = Vec2(x, y)
                        if tank_position_is_clear(point, radius, obs):
                            self.nodes.append(point)
            self.graph = [[] for _ in self.nodes]
            for i, first in enumerate(self.nodes):
                for j in range(i):
                    if self.clear(obs, first, self.nodes[j], radius):
                        length = (first - self.nodes[j]).length()
                        self.graph[i].append((j, length))
                        self.graph[j].append((i, length))
        start, goal = obs.me.position, obs.opponent.position
        if self.clear(obs, start, goal, self.rules.tank_radius):
            return goal
        queue, costs, next_node = [], {}, {}
        for i, point in enumerate(self.nodes):
            if self.clear(obs, point, goal, self.rules.tank_radius):
                costs[i] = (point - goal).length()
                heapq.heappush(queue, (costs[i], i))
        while queue:
            cost, i = heapq.heappop(queue)
            if cost != costs[i]:
                continue
            for j, length in self.graph[i]:
                if cost + length < costs.get(j, math.inf):
                    costs[j] = cost + length
                    next_node[j] = i
                    heapq.heappush(queue, (cost + length, j))
        choices = [
            (cost + (self.nodes[i] - start).length(), i)
            for i, cost in costs.items()
            if self.clear(obs, start, self.nodes[i], self.rules.tank_radius)
        ]
        if choices:
            return self.nodes[min(choices)[1]]
        # Rare narrow/rounded passage fallback to the repository's grid planner.
        from ricochet.bot import find_path

        path = find_path(obs, start, goal, self.rules.tank_radius, spacing=20)
        return next((p for p in path if (p - start).length() > 10), goal)

    def choose_move(self, obs):
        me, enemy, rules = obs.me, obs.opponent, self.rules
        distance = (enemy.position - me.position).length()
        if (
            obs.tick % 12 == 0
            or self.waypoint is None
            or (self.waypoint - me.position).length() < 14
        ):
            self.waypoint = self.navigation(obs)
        contact = self.enemy_shields and enemy.velocity.length() < 5 and distance < 110
        if contact:
            candidates = [
                enemy.position + direction * (2 * rules.tank_radius)
                for direction in (Vec2(1, 0), Vec2(-1, 0), Vec2(0, 1), Vec2(0, -1))
            ]
            candidates = [
                p
                for p in candidates
                if tank_position_is_clear(p, rules.tank_radius, obs)
                and self.clear(obs, me.position, p, rules.tank_radius)
            ]
            if candidates:
                self.waypoint = min(candidates, key=lambda p: (p - me.position).length())
        toward = (self.waypoint - me.position).normalized()
        # Standing still regenerates firing energy; constant movement is net energy-negative.
        preferred = toward if distance > 2 * rules.tank_radius + 0.01 else ZERO
        if (
            distance < 150
            and me.energy < rules.fire_energy
            and not (self.enemy_shields and enemy.velocity.length() < 5)
        ):
            preferred = ZERO
        if contact:
            preferred = (
                (self.waypoint - me.position) * (1 / (rules.tank_speed * rules.dt))
            ).limited()
        if self.short_range_shooter and not contact and distance < 190:
            # Stay far enough away to observe and evade a newly fired projectile.
            if distance < 150:
                retreat = (me.position - enemy.position).normalized()
                destination = me.position + retreat * 45
                if tank_position_is_clear(destination, rules.tank_radius, obs):
                    self.waypoint = destination
                    preferred = retreat
            else:
                self.waypoint = me.position
                preferred = ZERO
        paths = []
        horizon = 0.65
        for shot in obs.projectiles:
            if (shot.position - me.position).length() > (
                rules.projectile_speed + rules.tank_speed
            ) * horizon + 30:
                continue
            for segment in trajectory(
                shot.position,
                shot.velocity,
                shot.bounced,
                min(horizon, shot.remaining_lifetime),
                rules.projectile_radius,
                obs,
            ):
                if shot.owner_id != me.player_id or segment[4]:
                    paths.append(segment)
        if not paths:
            return preferred, False
        moves = [ZERO, preferred, self.previous_move]
        moves += [Vec2(math.cos(i * math.tau / 16), math.sin(i * math.tau / 16)) for i in range(16)]
        best, chosen, immediate = math.inf, ZERO, False
        for move in moves:
            position, energy, risk, urgent = me.position, me.energy, 0.0, False
            for step in range(10):
                t0, t1 = step * horizon / 10, (step + 1) * horizon / 10
                energy += rules.energy_regen_per_second * (t1 - t0)
                cost = rules.movement_energy_per_second * move.length() * (t1 - t0)
                fraction = min(1, energy / cost) if cost else 1
                energy = max(0, energy - cost)
                previous = position
                position = slide_circle(
                    position,
                    move * (rules.tank_speed * (t1 - t0) * fraction),
                    rules.tank_radius,
                    obs,
                )
                for a, b, p, q, _ in paths:
                    lo, hi = max(a, t0), min(b, t1)
                    if hi <= lo or b <= a:
                        continue
                    first = (
                        p
                        + (q - p) * ((lo - a) / (b - a))
                        - (previous + (position - previous) * ((lo - t0) / (t1 - t0)))
                    )
                    last = (
                        p
                        + (q - p) * ((hi - a) / (b - a))
                        - (previous + (position - previous) * ((hi - t0) / (t1 - t0)))
                    )
                    hit = segment_circle_time(
                        first, last, ZERO, rules.tank_radius + rules.projectile_radius + 3
                    )
                    if hit is not None:
                        risk += 2000 * (1 - 0.5 * lo / horizon)
                        urgent |= lo < rules.dt * 1.5
            progress = (position - self.waypoint).length() - (me.position - self.waypoint).length()
            score = risk + progress * 0.5 + move.length() * (18 if me.energy < 40 else 3)
            if distance < 40:
                score += (position - me.position).length() * 0.6
            score += (move - self.previous_move).length() * 0.3
            if score < best:
                best, chosen, immediate = score, move, urgent
        return chosen, immediate

    def shot(self, obs, origin):
        enemy, rules = obs.opponent, self.rules
        speed = rules.projectile_speed
        # Mirror constructions propose banks; exact swept collision validates the actual path.
        faces = [
            (0, rules.projectile_radius),
            (0, obs.width - rules.projectile_radius),
            (1, rules.projectile_radius),
            (1, obs.height - rules.projectile_radius),
        ]
        if self.use_banks:
            for wall in obs.walls:
                faces += [
                    (0, wall.left - rules.projectile_radius),
                    (0, wall.right + rules.projectile_radius),
                    (1, wall.top - rules.projectile_radius),
                    (1, wall.bottom + rules.projectile_radius),
                ]
        best = None
        for face in [None] + (faces if self.use_banks else []):
            target, velocity = enemy.position, enemy.velocity
            if face:
                axis, plane = face
                target = (
                    Vec2(2 * plane - target.x, target.y)
                    if axis == 0
                    else Vec2(target.x, 2 * plane - target.y)
                )
                velocity = (
                    Vec2(-velocity.x, velocity.y) if axis == 0 else Vec2(velocity.x, -velocity.y)
                )
            flight = intercept(target - origin, velocity, speed)
            if flight > min(2.5, rules.projectile_lifetime):
                continue
            predicted = enemy.position
            for _ in range(max(1, math.ceil(flight / 0.08))):
                predicted = slide_circle(
                    predicted,
                    enemy.velocity * (flight / max(1, math.ceil(flight / 0.08))),
                    rules.tank_radius,
                    obs,
                )
            target = predicted
            if face:
                target = (
                    Vec2(2 * plane - target.x, target.y)
                    if axis == 0
                    else Vec2(target.x, 2 * plane - target.y)
                )
            aim = (target - origin).normalized()
            segments = trajectory(
                origin,
                aim * speed,
                False,
                min(flight + 0.15, rules.projectile_lifetime),
                rules.projectile_radius,
                obs,
            )
            hit_time = None
            for a, b, p, q, bounced in segments:
                if (
                    bounced
                    and segment_circle_time(
                        p, q, origin, rules.tank_radius + rules.projectile_radius
                    )
                    is not None
                ):
                    break
                hit = segment_circle_time(
                    p, q, predicted, rules.tank_radius + rules.projectile_radius - 2
                )
                if hit is not None:
                    hit_time = a + (b - a) * hit
                    break
            if hit_time is not None:
                score = hit_time + (0.12 if face else 0)
                if best is None or score < best[0]:
                    best = (score, aim)
        return best

    def act(self, obs):
        rules, me = self.rules, obs.me
        self.enemy_shields |= obs.opponent.shield_active
        self.enemy_attacks |= any(p.owner_id == obs.opponent.player_id for p in obs.projectiles)
        enemy = obs.opponent
        approaching = enemy.velocity.dot((me.position - enemy.position).normalized())
        if approaching > 0.3 * rules.tank_speed and (enemy.position - me.position).length() < 300:
            self.short_range_shooter |= any(
                p.owner_id == enemy.player_id
                and (p.position - enemy.position).length()
                < rules.projectile_speed * rules.dt * 2 + 10
                for p in obs.projectiles
            )
        move, urgent = self.choose_move(obs)
        self.previous_move = move
        shield = (
            self.use_shield
            and urgent
            and me.energy
            >= (
                rules.shield_activation_cost
                + rules.shield_min_ticks * rules.shield_energy_per_second * rules.dt
            )
        )
        origin = slide_circle(
            me.position, move * (rules.tank_speed * rules.dt), rules.tank_radius, obs
        )
        available = (
            me.energy
            + rules.energy_regen_per_second * rules.dt
            - rules.movement_energy_per_second * rules.dt * move.length()
        )
        solution = None
        stationary_far = (
            self.enemy_shields
            and obs.opponent.velocity.length() < 5
            and (obs.opponent.position - origin).length() > 2 * rules.tank_radius + 0.01
        )
        if (
            not shield
            and not stationary_far
            and available >= rules.fire_energy
            and me.cooldown_remaining <= rules.dt
        ):
            solution = self.shot(obs, origin)
        aim = solution[1] if solution else obs.opponent.position - origin
        fire = solution is not None
        self.shots_fired += int(fire)
        return ControlAction(move=move, aim=aim, fire=fire, shield=shield)


def create_controller():
    return Sentinel()
