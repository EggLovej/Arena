from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from ricochet.game.config import RULES_VERSION, RulesConfig

PLAYER_ONE = 0
PLAYER_TWO = 1
PLAYERS = (PLAYER_ONE, PLAYER_TWO)


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self, fallback: Vec2 | None = None) -> Vec2:
        length = self.length()
        if length <= 1e-12:
            return fallback if fallback is not None else Vec2(0.0, 0.0)
        return Vec2(self.x / length, self.y / length)

    def limited(self, maximum: float = 1.0) -> Vec2:
        length = self.length()
        if length <= maximum:
            return self
        return self * (maximum / length)

    def to_list(self) -> list[float]:
        return [self.x, self.y]


@dataclass(frozen=True)
class ControlAction:
    move: Vec2 = Vec2(0.0, 0.0)
    aim: Vec2 = Vec2(0.0, 0.0)
    fire: bool = False
    shield: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "move": self.move.to_list(),
            "aim": self.aim.to_list(),
            "fire": self.fire,
            "shield": self.shield,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControlAction:
        return cls(
            move=Vec2(float(data["move"][0]), float(data["move"][1])),
            aim=Vec2(float(data["aim"][0]), float(data["aim"][1])),
            fire=bool(data.get("fire", False)),
            shield=bool(data.get("shield", False)),
        )


def create_noop_action() -> ControlAction:
    return ControlAction()


def validate_action(action: object) -> ControlAction:
    if not isinstance(action, ControlAction):
        raise ValueError("controller must return ControlAction")
    numbers = (action.move.x, action.move.y, action.aim.x, action.aim.y)
    if not all(math.isfinite(number) for number in numbers):
        raise ValueError("action vectors must contain finite numbers")
    if not isinstance(action.fire, bool) or not isinstance(action.shield, bool):
        raise ValueError("fire and shield must be booleans")
    return ControlAction(
        move=action.move.limited(),
        aim=action.aim.limited(),
        fire=action.fire,
        shield=action.shield,
    )


# Map geometry is always stored as floats, however it was constructed, so that captured
# state (and therefore every state hash) round-trips exactly through JSON.


@dataclass(frozen=True)
class WallDefinition:
    id: str
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id))
        for name in ("x", "y", "width", "height"):
            object.__setattr__(self, name, float(getattr(self, name)))

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass(frozen=True)
class SpawnDefinition:
    player_id: int
    position: Vec2
    aim: Vec2

    def __post_init__(self) -> None:
        object.__setattr__(self, "player_id", int(self.player_id))
        for name in ("position", "aim"):
            vector = getattr(self, name)
            object.__setattr__(self, name, Vec2(float(vector.x), float(vector.y)))


@dataclass(frozen=True)
class MapDefinition:
    """One serialized form for map files, captured engine state, replays and manifests."""

    id: str
    name: str
    width: float
    height: float
    spawns: tuple[SpawnDefinition, ...]
    walls: tuple[WallDefinition, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", float(self.width))
        object.__setattr__(self, "height", float(self.height))
        object.__setattr__(self, "spawns", tuple(self.spawns))
        object.__setattr__(self, "walls", tuple(self.walls))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MapDefinition:
        return cls(
            id=data["id"],
            name=data["name"],
            width=data["width"],
            height=data["height"],
            spawns=tuple(
                SpawnDefinition(spawn["player_id"], Vec2(**spawn["position"]), Vec2(**spawn["aim"]))
                for spawn in data["spawns"]
            ),
            walls=tuple(WallDefinition(**wall) for wall in data.get("walls", ())),
            schema_version=data.get("schema_version", 1),
        )


@dataclass(frozen=True)
class TankView:
    player_id: int
    position: Vec2
    velocity: Vec2
    aim: Vec2
    health: int
    energy: float
    cooldown_remaining: float
    shield_active: bool
    shield_ticks_remaining: int = 0


@dataclass(frozen=True)
class ProjectileView:
    id: int
    owner_id: int
    position: Vec2
    velocity: Vec2
    bounced: bool
    remaining_lifetime: float


@dataclass(frozen=True)
class GameObservation:
    player_id: int
    tick: int
    time_seconds: float
    max_seconds: float
    map_id: str
    width: float
    height: float
    tanks: tuple[TankView, ...]
    walls: tuple[WallDefinition, ...]
    projectiles: tuple[ProjectileView, ...]

    @property
    def me(self) -> TankView:
        return self.tank(self.player_id)

    @property
    def opponent(self) -> TankView:
        return self.tank(1 - self.player_id)

    def tank(self, player_id: int) -> TankView:
        return next(tank for tank in self.tanks if tank.player_id == player_id)


class EventKind:
    """Every ``GameEvent.kind``. Plain strings, so recorded events stay ordinary JSON."""

    FIRED = "fired"
    BOUNCED = "bounced"
    HIT = "hit"
    BLOCKED = "blocked"
    EXPIRED = "expired"
    REMOVED_AFTER_BOUNCE = "removed_after_bounce"
    SHIELD_ACTIVATED = "shield_activated"
    SHIELD_DEACTIVATED = "shield_deactivated"
    MATCH_ENDED = "match_ended"
    TIME_LIMIT = "time_limit"


@dataclass(frozen=True)
class GameEvent:
    kind: str
    tick: int
    player_id: int | None = None
    projectile_id: int | None = None
    value: float | int | str | None = None

    position: Vec2 | None = None
    normal: Vec2 | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.to_list() if self.position else None,
            "normal": self.normal.to_list() if self.normal else None,
            "kind": self.kind,
            "tick": self.tick,
            "player_id": self.player_id,
            "projectile_id": self.projectile_id,
            "value": self.value,
        }


@dataclass(frozen=True)
class MatchContext:
    map_id: str
    player_id: int
    opponent_id: int
    seed: int
    ticks_per_second: int
    max_ticks: int
    rules: RulesConfig = RulesConfig()
    rules_version: int = RULES_VERSION


@dataclass(frozen=True)
class StepResult:
    observations: tuple[GameObservation, GameObservation]
    events: tuple[GameEvent, ...]
    terminated: bool
    timed_out: bool
    winner: int | None
    state_hash: str

    def observation_for(self, player_id: int) -> GameObservation:
        return self.observations[player_id]


@dataclass(frozen=True)
class RenderSnapshot:
    observation: GameObservation
    rules: RulesConfig
    terminated: bool
    winner: int | None
