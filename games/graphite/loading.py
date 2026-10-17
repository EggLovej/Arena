"""Load and validate map files and AI modules."""

import importlib
import json
from collections import deque
from collections.abc import Callable
from pathlib import Path

from game_types import (
    BLUE,
    NEUTRAL,
    PLAYERS,
    RED,
    BaseDefinition,
    EdgeDefinition,
    GameView,
    MapDefinition,
    SendAction,
)


ROOT = Path(__file__).resolve().parent
MAPS_DIRECTORY = ROOT / "maps"
AIS_DIRECTORY = ROOT / "ais"

AiFunction = Callable[[GameView], SendAction | None]


def available_maps() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in MAPS_DIRECTORY.glob("*.json")))


def available_ais() -> tuple[str, ...]:
    return tuple(
        sorted(
            path.stem
            for path in AIS_DIRECTORY.glob("*.py")
            if not path.name.startswith("_")
        )
    )


def load_map(name: str) -> MapDefinition:
    if name not in available_maps():
        choices = ", ".join(available_maps())
        raise ValueError(f"Unknown map '{name}'. Available maps: {choices}")

    path = MAPS_DIRECTORY / f"{name}.json"
    data = json.loads(path.read_text())
    map_definition = MapDefinition(
        name=data.get("name", name),
        width=data["width"],
        height=data["height"],
        bases=tuple(BaseDefinition(**base) for base in data["bases"]),
        edges=tuple(EdgeDefinition(**edge) for edge in data["edges"]),
        time_limit=data.get("time_limit"),
        tagline=data.get("tagline", ""),
    )
    validate_map(map_definition)
    return map_definition


def validate_map(map_definition: MapDefinition) -> None:
    if map_definition.width <= 0 or map_definition.height <= 0:
        raise ValueError("Map width and height must be positive")
    if not map_definition.bases:
        raise ValueError("A map must contain at least one base")
    if map_definition.time_limit is not None and map_definition.time_limit <= 0:
        raise ValueError("A map's time limit must be positive")

    base_ids = [base.id for base in map_definition.bases]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError("Base IDs must be unique")

    for base in map_definition.bases:
        if base.owner not in (*PLAYERS, NEUTRAL):
            raise ValueError(f"Base '{base.id}' has an invalid owner")
        if base.units < 0 or base.production_rate < 0:
            raise ValueError(f"Base '{base.id}' has negative units or production")
        if not 0 <= base.x <= map_definition.width:
            raise ValueError(f"Base '{base.id}' is outside the map width")
        if not 0 <= base.y <= map_definition.height:
            raise ValueError(f"Base '{base.id}' is outside the map height")

    starting_owners = {base.owner for base in map_definition.bases}
    if RED not in starting_owners or BLUE not in starting_owners:
        raise ValueError("A map must contain starting bases for Red and Blue")

    known_ids = set(base_ids)
    edge_keys = set()
    adjacency = {base_id: set() for base_id in base_ids}
    for edge in map_definition.edges:
        if edge.a not in known_ids or edge.b not in known_ids:
            raise ValueError(f"Edge '{edge.a}-{edge.b}' references an unknown base")
        if edge.a == edge.b:
            raise ValueError("An edge cannot connect a base to itself")
        if edge.travel_time <= 0:
            raise ValueError("Edge travel times must be positive")

        key = frozenset((edge.a, edge.b))
        if key in edge_keys:
            raise ValueError(f"Duplicate edge: {edge.a}-{edge.b}")
        edge_keys.add(key)
        adjacency[edge.a].add(edge.b)
        adjacency[edge.b].add(edge.a)

    visited = set()
    queue = deque((base_ids[0],))
    while queue:
        base_id = queue.popleft()
        if base_id in visited:
            continue
        visited.add(base_id)
        queue.extend(adjacency[base_id] - visited)
    if visited != known_ids:
        raise ValueError("All bases must belong to one connected graph")


def load_ai(name: str) -> AiFunction:
    module_name = name.removesuffix(".py")
    if not module_name.isidentifier():
        raise ValueError("AI names must be simple Python module names")

    try:
        module = importlib.import_module(f"ais.{module_name}")
    except ModuleNotFoundError as error:
        if error.name == f"ais.{module_name}":
            raise ValueError(f"Could not find ais/{module_name}.py") from error
        raise

    choose_action = getattr(module, "choose_action", None)
    if not callable(choose_action):
        raise ValueError(f"ais/{module_name}.py must define choose_action(state)")
    return choose_action
