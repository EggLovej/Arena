from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from ricochet.game.config import RULES_VERSION, RulesConfig
from ricochet.game.maps import validate_map
from ricochet.game.physics import (
    EPSILON,
    earliest_wall_collision,
    reflect,
    resolve_tank_overlap,
    segment_circle_time,
    slide_circle,
)
from ricochet.game.types import (
    PLAYERS,
    ControlAction,
    EventKind,
    GameEvent,
    GameObservation,
    MapDefinition,
    MatchContext,
    ProjectileView,
    RenderSnapshot,
    StepResult,
    TankView,
    Vec2,
    create_noop_action,
    validate_action,
)


@dataclass
class _TankState:
    player_id: int
    position: Vec2
    velocity: Vec2
    aim: Vec2
    health: int
    energy: float
    cooldown_remaining: float = 0.0
    shield_active: bool = False
    shield_ticks_remaining: int = 0


@dataclass
class _ProjectileState:
    id: int
    owner_id: int
    position: Vec2
    velocity: Vec2
    bounced: bool
    remaining_lifetime: float


class GameEngine:
    """Deterministic fixed-step simulation with no display or training dependencies."""

    def __init__(
        self,
        map_definition: MapDefinition,
        rules: RulesConfig | None = None,
        seed: int = 0,
        verification: bool = True,
    ) -> None:
        self.verification = verification
        self._start(map_definition, rules or RulesConfig(), seed)

    @classmethod
    def from_state(cls, state: dict, verification: bool = True) -> GameEngine:
        """Build an engine entirely from ``capture_state`` output; no map file is needed."""
        engine = cls(
            MapDefinition.from_dict(state["map"]),
            RulesConfig.from_dict(state["rules"]),
            state["seed"],
            verification,
        )
        engine.restore_state(state)
        return engine

    def _start(self, map_definition: MapDefinition, rules: RulesConfig, seed: int) -> None:
        validate_map(map_definition, rules)
        self.map_definition = map_definition
        self.rules = rules
        self.seed = seed
        self.tick = 0
        self.tanks = {
            spawn.player_id: _TankState(
                player_id=spawn.player_id,
                position=spawn.position,
                velocity=Vec2(0, 0),
                aim=spawn.aim,
                health=self.rules.starting_health,
                energy=self.rules.max_energy,
            )
            for spawn in map_definition.spawns
        }
        self.projectiles: list[_ProjectileState] = []
        self._next_projectile_id = 0
        self.terminated = False
        self.timed_out = False
        self.winner: int | None = None

    @property
    def time_seconds(self) -> float:
        return self.tick * self.rules.dt

    def observation_for(self, player_id: int) -> GameObservation:
        tanks = tuple(
            TankView(
                player_id=tank.player_id,
                position=tank.position,
                velocity=tank.velocity,
                aim=tank.aim,
                health=tank.health,
                energy=tank.energy,
                cooldown_remaining=tank.cooldown_remaining,
                shield_active=tank.shield_active,
                shield_ticks_remaining=tank.shield_ticks_remaining,
            )
            for tank in (self.tanks[player] for player in PLAYERS)
        )
        projectiles = tuple(
            ProjectileView(
                id=projectile.id,
                owner_id=projectile.owner_id,
                position=projectile.position,
                velocity=projectile.velocity,
                bounced=projectile.bounced,
                remaining_lifetime=projectile.remaining_lifetime,
            )
            for projectile in sorted(self.projectiles, key=lambda item: item.id)
        )
        return GameObservation(
            player_id=player_id,
            tick=self.tick,
            time_seconds=self.time_seconds,
            max_seconds=self.rules.max_seconds,
            map_id=self.map_definition.id,
            width=self.map_definition.width,
            height=self.map_definition.height,
            tanks=tanks,
            walls=self.map_definition.walls,
            projectiles=projectiles,
        )

    def context_for(self, player_id: int) -> MatchContext:
        return MatchContext(
            map_id=self.map_definition.id,
            player_id=player_id,
            opponent_id=1 - player_id,
            seed=self.seed,
            ticks_per_second=self.rules.ticks_per_second,
            max_ticks=self.rules.max_ticks,
            rules=self.rules,
        )

    def observations(self) -> tuple[GameObservation, GameObservation]:
        return self.observation_for(0), self.observation_for(1)

    def step(self, actions: dict[int, ControlAction]) -> StepResult:
        if self.terminated or self.timed_out:
            raise RuntimeError("cannot step a finished game")
        normalized = {
            player_id: validate_action(actions.get(player_id, create_noop_action()))
            for player_id in PLAYERS
        }
        events: list[GameEvent] = []
        self._prepare_tanks(normalized, events)
        self._move_tanks(normalized)
        self._fire(normalized, events)
        self._move_projectiles(events)
        self.tick += 1
        self._finish_if_needed(events)
        return StepResult(
            observations=self.observations(),
            events=tuple(events),
            terminated=self.terminated,
            timed_out=self.timed_out,
            winner=self.winner,
            state_hash=self.state_hash() if self.verification else "",
        )

    def snapshot(self) -> StepResult:
        return StepResult(
            observations=self.observations(),
            events=(),
            terminated=self.terminated,
            timed_out=self.timed_out,
            winner=self.winner,
            state_hash=self.state_hash() if self.verification else "",
        )

    def _prepare_tanks(self, actions: dict[int, ControlAction], events: list[GameEvent]) -> None:
        dt = self.rules.dt
        for player_id in PLAYERS:
            tank = self.tanks[player_id]
            tank.energy = min(
                self.rules.max_energy,
                tank.energy + self.rules.energy_regen_per_second * dt,
            )
            tank.cooldown_remaining = max(0.0, tank.cooldown_remaining - dt)
            requested_shield = actions[player_id].shield
            shield_cost = self.rules.shield_energy_per_second * dt
            was_active = tank.shield_active
            if tank.shield_ticks_remaining > 0:
                tank.shield_active = True
                tank.shield_ticks_remaining -= 1
            elif was_active and requested_shield and tank.energy + EPSILON >= shield_cost:
                tank.energy = max(0.0, tank.energy - shield_cost)
                tank.shield_active = True
            elif not was_active and requested_shield:
                ticks = self.rules.shield_min_ticks
                upfront = self.rules.shield_activation_cost + ticks * shield_cost
                tank.shield_active = tank.energy + EPSILON >= upfront
                if tank.shield_active:
                    tank.energy = max(0.0, tank.energy - upfront)
                    tank.shield_ticks_remaining = ticks - 1
            else:
                tank.shield_active = False
            if was_active != tank.shield_active:
                kind = (
                    EventKind.SHIELD_ACTIVATED
                    if tank.shield_active
                    else EventKind.SHIELD_DEACTIVATED
                )
                events.append(GameEvent(kind, self.tick, player_id, position=tank.position))
            aim = actions[player_id].aim
            if aim.length() > EPSILON:
                tank.aim = aim.normalized()

    def _move_tanks(self, actions: dict[int, ControlAction]) -> None:
        dt = self.rules.dt
        old_positions = {player_id: self.tanks[player_id].position for player_id in PLAYERS}
        proposed: dict[int, Vec2] = {}
        for player_id in PLAYERS:
            tank = self.tanks[player_id]
            movement = actions[player_id].move.limited()
            magnitude = movement.length()
            move_cost = self.rules.movement_energy_per_second * dt * magnitude
            if move_cost > tank.energy and move_cost > 0:
                movement = movement * (tank.energy / move_cost)
                move_cost = tank.energy
            tank.energy -= move_cost
            speed = self.rules.tank_speed
            if tank.shield_active:
                speed *= self.rules.shield_speed_multiplier
            displacement = movement * (speed * dt)
            proposed[player_id] = slide_circle(
                tank.position,
                displacement,
                self.rules.tank_radius,
                self.map_definition,
            )
        first, second = resolve_tank_overlap(
            proposed[0],
            proposed[1],
            self.rules.tank_radius,
            self.map_definition,
        )
        if (second - first).length() < 2 * self.rules.tank_radius - EPSILON:
            first, second = old_positions[0], old_positions[1]
        proposed = {0: first, 1: second}
        for player_id in PLAYERS:
            tank = self.tanks[player_id]
            tank.position = proposed[player_id]
            tank.velocity = (tank.position - old_positions[player_id]) * (1.0 / dt)

    def _fire(self, actions: dict[int, ControlAction], events: list[GameEvent]) -> None:
        for player_id in PLAYERS:
            tank = self.tanks[player_id]
            action = actions[player_id]
            can_fire = (
                action.fire
                and not tank.shield_active
                and tank.health > 0
                and tank.cooldown_remaining <= EPSILON
                and tank.energy + EPSILON >= self.rules.fire_energy
                and len(self.projectiles) < self.rules.max_projectiles
            )
            if not can_fire:
                continue
            tank.energy -= self.rules.fire_energy
            tank.cooldown_remaining = self.rules.fire_cooldown_seconds
            projectile = _ProjectileState(
                id=self._next_projectile_id,
                owner_id=player_id,
                position=tank.position,
                velocity=tank.aim * self.rules.projectile_speed,
                bounced=False,
                remaining_lifetime=self.rules.projectile_lifetime,
            )
            self._next_projectile_id += 1
            self.projectiles.append(projectile)
            events.append(
                GameEvent(
                    EventKind.FIRED, self.tick, player_id, projectile.id, position=tank.position
                )
            )

    def _move_projectiles(self, events: list[GameEvent]) -> None:
        survivors: list[_ProjectileState] = []
        damage = {0: 0, 1: 0}
        for projectile in sorted(self.projectiles, key=lambda item: item.id):
            projectile.remaining_lifetime -= self.rules.dt
            if projectile.remaining_lifetime <= 0:
                events.append(
                    GameEvent(EventKind.EXPIRED, self.tick, projectile.owner_id, projectile.id)
                )
                continue
            if self._advance_projectile(projectile, damage, events):
                survivors.append(projectile)
        self.projectiles = survivors
        for player_id in PLAYERS:
            if damage[player_id]:
                self.tanks[player_id].health = max(
                    0,
                    self.tanks[player_id].health - damage[player_id],
                )

    def _advance_projectile(
        self,
        projectile: _ProjectileState,
        damage: dict[int, int],
        events: list[GameEvent],
    ) -> bool:
        remaining = self.rules.dt
        for _ in range(3):
            start = projectile.position
            end = start + projectile.velocity * remaining
            wall_collision = earliest_wall_collision(
                start,
                end,
                self.rules.projectile_radius,
                self.map_definition,
            )
            tank_collision: tuple[float, int] | None = None
            for player_id in PLAYERS:
                if player_id == projectile.owner_id and not projectile.bounced:
                    continue
                tank = self.tanks[player_id]
                if tank.health <= 0:
                    continue
                time = segment_circle_time(
                    start,
                    end,
                    tank.position,
                    self.rules.tank_radius + self.rules.projectile_radius,
                )
                if time is not None and (
                    tank_collision is None or (time, player_id) < tank_collision
                ):
                    tank_collision = (time, player_id)
            if tank_collision is not None and (
                wall_collision is None or tank_collision[0] <= wall_collision.t + EPSILON
            ):
                hit_player = tank_collision[1]
                contact = start + (end - start) * tank_collision[0]
                if self.tanks[hit_player].shield_active:
                    events.append(
                        GameEvent(
                            EventKind.BLOCKED,
                            self.tick,
                            hit_player,
                            projectile.id,
                            projectile.owner_id,
                            position=contact,
                        )
                    )
                else:
                    damage[hit_player] += 1
                    events.append(
                        GameEvent(
                            EventKind.HIT,
                            self.tick,
                            hit_player,
                            projectile.id,
                            projectile.owner_id,
                            position=contact,
                        )
                    )
                return False
            if wall_collision is None:
                projectile.position = end
                return True
            collision_t = max(0.0, min(1.0, wall_collision.t))
            projectile.position = start + (end - start) * collision_t
            if projectile.bounced:
                events.append(
                    GameEvent(
                        EventKind.REMOVED_AFTER_BOUNCE,
                        self.tick,
                        projectile.owner_id,
                        projectile.id,
                    )
                )
                return False
            projectile.velocity = reflect(projectile.velocity, wall_collision.normal)
            projectile.bounced = True
            projectile.position = projectile.position + wall_collision.normal * 1e-5
            events.append(
                GameEvent(
                    EventKind.BOUNCED,
                    self.tick,
                    projectile.owner_id,
                    projectile.id,
                    position=projectile.position,
                    normal=wall_collision.normal,
                )
            )
            remaining *= max(0.0, 1.0 - collision_t)
            if remaining <= EPSILON:
                return True
        return True

    def _finish_if_needed(self, events: list[GameEvent]) -> None:
        alive = [player_id for player_id in PLAYERS if self.tanks[player_id].health > 0]
        if len(alive) < 2:
            self.terminated = True
            self.winner = alive[0] if len(alive) == 1 else None
            events.append(GameEvent(EventKind.MATCH_ENDED, self.tick, self.winner))
        elif self.tick >= self.rules.max_ticks:
            self.timed_out = True
            self.terminated = True
            self.winner = None
            events.append(GameEvent(EventKind.TIME_LIMIT, self.tick))

    def force_forfeit(self, losing_player: int) -> None:
        if losing_player not in PLAYERS:
            raise ValueError("unknown player")
        self.terminated = True
        self.winner = 1 - losing_player

    def render_snapshot(self) -> RenderSnapshot:
        return RenderSnapshot(self.observation_for(0), self.rules, self.terminated, self.winner)

    def capture_state(self) -> dict:
        return {
            "rules_version": RULES_VERSION,
            "map": self.map_definition.to_dict(),
            "rules": self.rules.to_dict(),
            "seed": self.seed,
            "tick": self.tick,
            "terminated": self.terminated,
            "timed_out": self.timed_out,
            "winner": self.winner,
            "next_projectile_id": self._next_projectile_id,
            "tanks": [asdict(self.tanks[i]) for i in PLAYERS],
            "projectiles": [asdict(p) for p in self.projectiles],
        }

    def restore_state(self, state: dict) -> None:
        if state["rules_version"] != RULES_VERSION:
            raise ValueError("incompatible rules version")
        self._start(
            MapDefinition.from_dict(state["map"]),
            RulesConfig.from_dict(state["rules"]),
            state["seed"],
        )
        self.tick = state["tick"]
        self.terminated, self.timed_out = state["terminated"], state["timed_out"]
        self.winner = state["winner"]
        self._next_projectile_id = state["next_projectile_id"]
        for tank in state["tanks"]:
            vectors = {name: Vec2(**tank[name]) for name in ("position", "velocity", "aim")}
            self.tanks[tank["player_id"]] = _TankState(**(tank | vectors))
        self.projectiles = [
            _ProjectileState(
                **(
                    projectile
                    | {name: Vec2(**projectile[name]) for name in ("position", "velocity")}
                )
            )
            for projectile in state["projectiles"]
        ]

    def clone(self) -> GameEngine:
        return GameEngine.from_state(self.capture_state(), self.verification)

    def state_hash(self) -> str:
        payload = json.dumps(self.capture_state(), separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()
