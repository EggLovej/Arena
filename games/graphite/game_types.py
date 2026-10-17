"""Public data types shared by the game and AI files."""

from dataclasses import dataclass


RED = -1
NEUTRAL = 0
BLUE = 1
GREEN = 2
PLAYERS = (RED, BLUE, GREEN)


@dataclass(frozen=True)
class BaseDefinition:
    id: str
    owner: int
    units: float
    production_rate: float
    x: int
    y: int


@dataclass(frozen=True)
class EdgeDefinition:
    a: str
    b: str
    travel_time: float


@dataclass(frozen=True)
class MapDefinition:
    name: str
    width: int
    height: int
    bases: tuple[BaseDefinition, ...]
    edges: tuple[EdgeDefinition, ...]
    # Seconds of game time before the match is a draw, for maps that take longer than
    # the usual three minutes to play out. None keeps the usual limit.
    time_limit: float | None = None
    tagline: str = ""  # a few words on the map's character, for people choosing a map


@dataclass(frozen=True)
class BaseView:
    id: str
    owner: int
    units: float
    production_rate: float
    x: int
    y: int


@dataclass(frozen=True)
class EdgeView:
    a: str
    b: str
    travel_time: float


@dataclass(frozen=True)
class FleetView:
    owner: int
    units: int
    source_id: str
    target_id: str
    remaining_time: float
    total_time: float


@dataclass(frozen=True)
class SendAction:
    """Send an integer number of units over one edge."""

    source_id: str
    target_id: str
    units: int


@dataclass(frozen=True)
class GameView:
    """An immutable, fully observable snapshot given to one AI."""

    player_id: int
    tick: int
    time_seconds: float
    seconds_per_tick: float
    bases: tuple[BaseView, ...]
    edges: tuple[EdgeView, ...]
    fleets: tuple[FleetView, ...]

    def get_base(self, base_id: str) -> BaseView:
        for base in self.bases:
            if base.id == base_id:
                return base
        raise KeyError(f"Unknown base: {base_id}")

    def my_bases(self) -> tuple[BaseView, ...]:
        return tuple(base for base in self.bases if base.owner == self.player_id)

    def neighbors(self, base_id: str) -> tuple[BaseView, ...]:
        neighbor_ids = []
        for edge in self.edges:
            if edge.a == base_id:
                neighbor_ids.append(edge.b)
            elif edge.b == base_id:
                neighbor_ids.append(edge.a)
        return tuple(self.get_base(neighbor_id) for neighbor_id in neighbor_ids)

    def travel_time(self, a: str, b: str) -> float:
        for edge in self.edges:
            if {edge.a, edge.b} == {a, b}:
                return edge.travel_time
        raise KeyError(f"There is no edge between {a} and {b}")
