"""Random 1v1 map generator.

Every map is built in the same steps, each a small function below:

1. place bases on the left half, mirror them to the right, add centre bases
2. connect them with routes that do not cross
3. choose starting bases
4. draw units and production for every base
5. optionally break the mirror symmetry by a controlled amount

Every tunable number lives in ``MapGenParams``. The same seed and parameters
always produce the same map.
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from game_types import (
    BLUE,
    NEUTRAL,
    RED,
    BaseDefinition,
    EdgeDefinition,
    MapDefinition,
)
from loading import MAPS_DIRECTORY, validate_map


@dataclass(frozen=True)
class MapGenParams:
    """Every number the generator uses.

    Edit a default here, or make a variant without touching this file:
    ``replace(DEFAULT_PARAMS, max_production=5)``.
    """

    # ---- Size -------------------------------------------------------------
    # A map has this many bases on each side of the mirror line, plus a few
    # bases that sit on the line itself and belong to neither side.
    min_bases_per_side: int = 2
    max_bases_per_side: int = 13
    max_centre_bases: int = 3
    # Each player starts with one base. Each further starting base is added
    # with this chance, up to the maximum.
    max_starting_bases: int = 2
    extra_starting_base_chance: float = 0.25

    # ---- Layout, in pixels --------------------------------------------------
    width: int = 1000
    height: int = 800
    # Bases stay inside this frame so the header, footer, and labels fit.
    field_margin_x: int = 90
    field_top: int = 140
    field_bottom: int = 670
    # Bases are spaced out as far as the map allows, between these two limits.
    min_base_distance: int = 85
    preferred_base_distance: int = 150
    # A route may not pass closer than this to a base it does not connect.
    min_route_clearance: int = 50

    # ---- Routes ---------------------------------------------------------------
    # Routes per base decides how connected a map is: 1.0 is a bare tree,
    # 2.0 is a dense mesh. Each map draws one value in this range.
    min_routes_per_base: float = 1.3
    max_routes_per_base: float = 1.9
    max_routes_at_one_base: int = 6
    # Travel time is route length divided by speed; each map draws one speed.
    # A map whose longest route would exceed the maximum is sped up to fit.
    min_travel_time: float = 1.0
    max_travel_time: float = 8.0
    min_pixels_per_second: float = 35.0
    max_pixels_per_second: float = 100.0

    # ---- Values: hard limits ------------------------------------------------
    min_production: float = 0.0
    max_production: float = 10.0
    min_starting_units: int = 0
    max_starting_units: int = 250
    min_neutral_units: int = 1
    max_neutral_units: int = 250

    # ---- Values: what is common inside those limits -------------------------
    # Half of all draws land below the typical value and half above it.
    typical_production: float = 1.2
    typical_starting_production: float = 1.2
    typical_starting_units: float = 25.0
    typical_neutral_units: float = 15.0
    # Spread works by multiplication: 0.5 means about two thirds of draws land
    # between 0.6x and 1.65x the typical value. 0 means no variation.
    map_spread: float = 0.5  # how much whole maps differ from each other
    base_spread: float = 0.5  # how much bases differ within one map
    # 0 = what a neutral base costs is unrelated to what it produces,
    # 1 = richer bases always cost more.
    cost_follows_production: float = 0.6
    production_step: float = 0.05

    # ---- Playability ----------------------------------------------------------
    # Each player must be able to afford a first capture within this long.
    max_opening_wait_seconds: float = 30.0

    # ---- Asymmetry --------------------------------------------------------------
    # This share of maps are perfect mirrors. The rest draw an asymmetry level
    # between 0 and the maximum.
    mirrored_share: float = 0.5
    max_asymmetry: float = 0.3
    # What asymmetry 1.0 means; lower levels scale these down in proportion.
    asymmetry_shift_pixels: float = 120.0  # how far a base may move
    asymmetry_value_spread: float = 1.0  # spread applied to units and production
    asymmetry_route_drop_chance: float = 0.3  # chance that a route is removed

    def __post_init__(self) -> None:
        ranges = (
            ("bases_per_side", self.min_bases_per_side, self.max_bases_per_side),
            ("routes_per_base", self.min_routes_per_base, self.max_routes_per_base),
            ("travel_time", self.min_travel_time, self.max_travel_time),
            ("pixels_per_second", self.min_pixels_per_second, self.max_pixels_per_second),
            ("production", self.min_production, self.max_production),
            ("starting_units", self.min_starting_units, self.max_starting_units),
            ("neutral_units", self.min_neutral_units, self.max_neutral_units),
            ("base_distance", self.min_base_distance, self.preferred_base_distance),
        )
        for name, low, high in ranges:
            if low > high:
                raise ValueError(f"min {name} is larger than max {name}")
        if self.min_bases_per_side < 1 or self.max_starting_bases < 1:
            raise ValueError("A map needs at least one base per side")
        if self.min_travel_time <= 0 or self.min_pixels_per_second <= 0:
            raise ValueError("Travel times and speeds must be positive")
        if self.min_production < 0 or self.min_starting_units < 0 or self.min_neutral_units < 0:
            raise ValueError("Units and production cannot be negative")
        if not 0 <= self.cost_follows_production <= 1:
            raise ValueError("cost_follows_production must be between 0 and 1")


DEFAULT_PARAMS = MapGenParams()


@dataclass(frozen=True)
class GeneratedMap:
    seed: int
    asymmetry: float
    definition: MapDefinition


# Internal knobs that shape the search, not the maps.
_MAX_ATTEMPTS = 200
_PLACEMENT_TRIES = 300
_VALUE_TRIES = 20
_SPACING_FACTOR = 0.8
_ROUTE_ORDER_JITTER = 1.3

_BASE_NAMES = (
    "alder", "amber", "aspen", "basalt", "birch", "briar", "cedar", "clay",
    "cobalt", "dune", "elm", "ember", "fern", "flint", "gorse", "grove",
    "hazel", "heath", "iris", "ivy", "jade", "juniper", "kelp", "larch",
    "lichen", "marl", "moss", "nettle", "oak", "ochre", "onyx", "peat",
    "pine", "quartz", "reed", "rowan", "sage", "slate", "thorn", "umber",
    "vale", "willow", "wren", "yarrow", "zinc",
)  # fmt: skip

Point = tuple[int, int]
Route = tuple[int, int]


@dataclass
class _Draft:
    """A map under construction. Bases are plain indexes until the very end."""

    positions: list[Point]
    mirror: list[int]  # index of each base's mirror image; centre bases map to themselves
    routes: set[Route]
    owners: list[int]
    units: list[int]
    production: list[float]
    pixels_per_second: float


def generate_map(
    seed: int,
    params: MapGenParams = DEFAULT_PARAMS,
    asymmetry: float | None = None,
) -> GeneratedMap:
    """Build one map. Leave ``asymmetry`` unset to draw it from the usual mix."""

    # Separate random streams, so one seed gives the same underlying map at
    # every asymmetry level.
    layout_rng = random.Random(f"{seed}:layout")
    asymmetry_rng = random.Random(f"{seed}:asymmetry")

    if asymmetry is None:
        name = f"generated_{seed}"
        asymmetry = _draw_asymmetry(random.Random(f"{seed}:mix"), params)
    else:
        name = f"generated_{seed}_a{round(asymmetry * 100):03d}"

    for _ in range(_MAX_ATTEMPTS):
        draft = _mirrored_draft(layout_rng, params)
        if draft is None:
            continue
        if asymmetry > 0:
            _break_symmetry(asymmetry_rng, params, draft, asymmetry)
            if not _opening_is_playable(draft, params):
                continue

        definition = _to_definition(layout_rng, params, draft, name)
        validate_map(definition)
        return GeneratedMap(seed=seed, asymmetry=asymmetry, definition=definition)

    raise RuntimeError(f"Could not build a map for seed {seed} with these parameters")


def map_to_json(map_definition: MapDefinition) -> str:
    """Serialize in the same layout as the hand-written files in ``maps/``."""

    bases = ",\n".join(
        "    " + json.dumps(
            {
                "id": base.id,
                "owner": base.owner,
                "units": base.units,
                "production_rate": base.production_rate,
                "x": base.x,
                "y": base.y,
            }
        )
        for base in map_definition.bases
    )
    edges = ",\n".join(
        "    " + json.dumps({"a": edge.a, "b": edge.b, "travel_time": edge.travel_time})
        for edge in map_definition.edges
    )
    return (
        "{\n"
        f'  "name": {json.dumps(map_definition.name)},\n'
        f'  "width": {map_definition.width},\n'
        f'  "height": {map_definition.height},\n'
        f'  "bases": [\n{bases}\n  ],\n'
        f'  "edges": [\n{edges}\n  ]\n'
        "}\n"
    )


def save_map(map_definition: MapDefinition, directory: Path = MAPS_DIRECTORY) -> Path:
    """Write the map where the game looks for maps, so it can be played."""

    path = Path(directory) / f"{map_definition.name}.json"
    path.write_text(map_to_json(map_definition))
    return path


def _draw_asymmetry(rng: random.Random, params: MapGenParams) -> float:
    if rng.random() < params.mirrored_share:
        return 0.0
    return rng.uniform(0.0, params.max_asymmetry)


def _mirrored_draft(rng: random.Random, params: MapGenParams) -> _Draft | None:
    """Steps 1–4: a perfectly mirrored map, or None if this attempt got stuck."""

    placed = _place_bases(rng, params)
    if placed is None:
        return None
    positions, mirror, side_count = placed

    routes = _build_routes(rng, params, positions, mirror)
    if routes is None:
        return None

    pixels_per_second = rng.uniform(params.min_pixels_per_second, params.max_pixels_per_second)
    longest = max(math.dist(positions[a], positions[b]) for a, b in routes)
    pixels_per_second = max(pixels_per_second, longest / params.max_travel_time)

    owners = _choose_starting_bases(rng, params, positions, mirror, side_count)
    draft = _Draft(positions, mirror, routes, owners, [], [], pixels_per_second)

    for _ in range(_VALUE_TRIES):
        draft.units, draft.production = _draw_values(rng, params, owners, mirror)
        if _opening_is_playable(draft, params):
            return draft
    return None


def _place_bases(
    rng: random.Random,
    params: MapGenParams,
) -> tuple[list[Point], list[int], int] | None:
    """Step 1. Scatter bases on the left, copy them to the right."""

    side_count = rng.randint(params.min_bases_per_side, params.max_bases_per_side)
    centre_count = rng.randint(0, params.max_centre_bases)
    axis = params.width // 2

    # Small maps get roomy spacing; crowded maps pack closer, down to the minimum.
    half_area = (axis - params.field_margin_x) * (params.field_bottom - params.field_top)
    spacing = _SPACING_FACTOR * math.sqrt(half_area / (side_count + centre_count / 2))
    spacing = min(max(spacing, params.min_base_distance), params.preferred_base_distance)

    left_edge = params.field_margin_x
    # Staying half a spacing away from the line keeps a base clear of its own mirror image.
    right_edge = axis - math.ceil(spacing / 2)
    if right_edge < left_edge:
        return None

    placed: list[Point] = []
    for index in range(centre_count + side_count):
        on_axis = index < centre_count
        for _ in range(_PLACEMENT_TRIES):
            x = axis if on_axis else rng.randint(left_edge, right_edge)
            y = rng.randint(params.field_top, params.field_bottom)
            if all(math.dist((x, y), other) >= spacing for other in placed):
                placed.append((x, y))
                break
        else:
            return None

    # Order: centre bases, then the left side, then the right side.
    left = placed[centre_count:]
    right = [(params.width - x, y) for x, y in left]
    positions = placed + right

    mirror = list(range(centre_count))
    mirror += [index + side_count for index in range(centre_count, centre_count + side_count)]
    mirror += list(range(centre_count, centre_count + side_count))
    return positions, mirror, side_count


def _build_routes(
    rng: random.Random,
    params: MapGenParams,
    positions: list[Point],
    mirror: list[int],
) -> set[Route] | None:
    """Step 2. Short routes first, always adding a route together with its mirror image."""

    count = len(positions)

    # A candidate is a route plus its mirror image (one route if it mirrors onto itself).
    candidates = []
    seen = set()
    for a in range(count):
        for b in range(a + 1, count):
            route = (a, b)
            pair = tuple(sorted({route, _route(mirror[a], mirror[b])}))
            if pair in seen:
                continue
            seen.add(pair)

            length = math.dist(positions[a], positions[b])
            if len(pair) == 2 and _routes_cross(positions, pair[0], pair[1]):
                continue
            if not _route_is_clear(positions, route, params.min_route_clearance):
                continue
            # The jitter varies which routes win, so maps are not all nearest-neighbour meshes.
            candidates.append((length * rng.uniform(1.0, _ROUTE_ORDER_JITTER), pair))
    candidates.sort()

    routes: set[Route] = set()
    degree = [0] * count

    def try_add(pair: tuple[Route, ...]) -> bool:
        added = [0] * count
        for a, b in pair:
            added[a] += 1
            added[b] += 1
        if any(degree[i] + added[i] > params.max_routes_at_one_base for i in range(count)):
            return False
        if any(_routes_cross(positions, new, old) for new in pair for old in routes):
            return False
        routes.update(pair)
        for i in range(count):
            degree[i] += added[i]
        return True

    # First pass: only routes that join two separate groups, until everything is connected.
    group = list(range(count))

    def find(i: int) -> int:
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i

    spare = []
    for _, pair in candidates:
        if any(find(a) != find(b) for a, b in pair) and try_add(pair):
            for a, b in pair:
                group[find(a)] = find(b)
        else:
            spare.append(pair)
    if len({find(i) for i in range(count)}) > 1:
        return None

    # Second pass: extra routes, shortest first, up to this map's target density.
    routes_per_base = rng.uniform(params.min_routes_per_base, params.max_routes_per_base)
    target = round(routes_per_base * count)
    for pair in spare:
        if len(routes) >= target:
            break
        try_add(pair)
    return routes


def _choose_starting_bases(
    rng: random.Random,
    params: MapGenParams,
    positions: list[Point],
    mirror: list[int],
    side_count: int,
) -> list[int]:
    """Step 3. Red starts on the left, far from the mirror line; Blue mirrors Red."""

    centre_count = len(positions) - 2 * side_count
    start_count = 1
    while (
        start_count < params.max_starting_bases
        and rng.random() < params.extra_starting_base_chance
    ):
        start_count += 1
    # Always leave at least one base to capture.
    if centre_count == 0:
        start_count = min(start_count, side_count - 1)
    start_count = max(1, min(start_count, side_count))

    left = sorted(range(centre_count, centre_count + side_count), key=lambda i: positions[i])
    outermost = left[: max(start_count, math.ceil(side_count / 3))]

    owners = [NEUTRAL] * len(positions)
    for index in rng.sample(outermost, start_count):
        owners[index] = RED
        owners[mirror[index]] = BLUE
    return owners


def _draw_values(
    rng: random.Random,
    params: MapGenParams,
    owners: list[int],
    mirror: list[int],
) -> tuple[list[int], list[float]]:
    """Step 4. Units and production for every base, identical on both sides."""

    # One scale per map, so some maps are rich or expensive throughout.
    unit_scale = math.exp(rng.gauss(0, params.map_spread))
    production_scale = math.exp(rng.gauss(0, params.map_spread))
    tie = params.cost_follows_production

    units = [0] * len(owners)
    production = [0.0] * len(owners)
    for index, owner in enumerate(owners):
        if mirror[index] < index:
            units[index] = units[mirror[index]]
            production[index] = production[mirror[index]]
            continue

        rich = rng.gauss(0, 1)
        costly = tie * rich + math.sqrt(1 - tie * tie) * rng.gauss(0, 1)
        if owner == NEUTRAL:
            production[index] = _production(
                params,
                params.typical_production * production_scale * math.exp(params.base_spread * rich),
            )
            units[index] = _neutral_units(
                params,
                params.typical_neutral_units * unit_scale * math.exp(params.base_spread * costly),
            )
        else:
            production[index] = _production(
                params,
                params.typical_starting_production
                * production_scale
                * math.exp(params.base_spread * rich),
            )
            units[index] = _starting_units(
                params,
                params.typical_starting_units
                * unit_scale
                * math.exp(params.base_spread * rng.gauss(0, 1)),
            )
    return units, production


def _break_symmetry(
    rng: random.Random,
    params: MapGenParams,
    draft: _Draft,
    asymmetry: float,
) -> None:
    """Step 5. Nudge bases, values, and routes on both sides independently."""

    positions = draft.positions
    count = len(positions)

    # Every base draws one direction; if the full move does not fit, try shorter ones.
    reach = asymmetry * params.asymmetry_shift_pixels
    for index in range(count):
        angle = rng.uniform(0, 2 * math.pi)
        distance = reach * math.sqrt(rng.random())
        home = positions[index]
        for scale in (1.0, 0.5, 0.25):
            positions[index] = (
                round(home[0] + math.cos(angle) * distance * scale),
                round(home[1] + math.sin(angle) * distance * scale),
            )
            if _position_is_valid(params, draft, index):
                break
        else:
            positions[index] = home

    spread = asymmetry * params.asymmetry_value_spread
    for index in range(count):
        production = draft.production[index] * math.exp(rng.gauss(0, spread))
        units = draft.units[index] * math.exp(rng.gauss(0, spread))
        draft.production[index] = _production(params, production)
        if draft.owners[index] == NEUTRAL:
            draft.units[index] = _neutral_units(params, units)
        else:
            draft.units[index] = _starting_units(params, units)

    drop_chance = asymmetry * params.asymmetry_route_drop_chance
    for route in sorted(draft.routes):
        if rng.random() < drop_chance and _connected(count, draft.routes - {route}):
            draft.routes.discard(route)


def _position_is_valid(params: MapGenParams, draft: _Draft, moved: int) -> bool:
    """Check one moved base: inside the frame, not crowding, no crossed routes."""

    positions = draft.positions
    x, y = positions[moved]
    if not params.field_margin_x <= x <= params.width - params.field_margin_x:
        return False
    if not params.field_top <= y <= params.field_bottom:
        return False
    if any(
        math.dist(positions[moved], positions[other]) < params.min_base_distance
        for other in range(len(positions))
        if other != moved
    ):
        return False

    for route in draft.routes:
        if moved in route:
            if not _route_is_clear(positions, route, params.min_route_clearance):
                return False
            if any(_routes_cross(positions, route, other) for other in draft.routes):
                return False
        elif _distance_to_route(positions, moved, route) < params.min_route_clearance:
            return False
    return True


def _opening_is_playable(draft: _Draft, params: MapGenParams) -> bool:
    return all(
        _opening_wait(draft, player) <= params.max_opening_wait_seconds
        for player in (RED, BLUE)
    )


def _opening_wait(draft: _Draft, player: int) -> float:
    """Seconds until the player can afford to capture some neighbouring neutral base."""

    shortest = math.inf
    for a, b in draft.routes:
        for source, target in ((a, b), (b, a)):
            if draft.owners[source] != player or draft.owners[target] != NEUTRAL:
                continue
            shortfall = draft.units[target] + 1 - draft.units[source]
            if shortfall <= 0:
                return 0.0
            if draft.production[source] > 0:
                shortest = min(shortest, shortfall / draft.production[source])
    return shortest


def _to_definition(
    rng: random.Random,
    params: MapGenParams,
    draft: _Draft,
    name: str,
) -> MapDefinition:
    """Give bases names, shuffle the order, and turn route lengths into travel times."""

    count = len(draft.positions)
    if count <= len(_BASE_NAMES):
        names = rng.sample(_BASE_NAMES, count)
    else:
        names = [f"base_{index:02d}" for index in range(count)]
    # Neither names nor list order may hint at a base's role.
    order = list(range(count))
    rng.shuffle(order)

    bases = tuple(
        BaseDefinition(
            id=names[index],
            owner=draft.owners[index],
            units=draft.units[index],
            production_rate=draft.production[index],
            x=draft.positions[index][0],
            y=draft.positions[index][1],
        )
        for index in order
    )

    routes = sorted(draft.routes)
    rng.shuffle(routes)
    edges = []
    for a, b in routes:
        seconds = math.dist(draft.positions[a], draft.positions[b]) / draft.pixels_per_second
        seconds = min(max(seconds, params.min_travel_time), params.max_travel_time)
        edges.append(EdgeDefinition(a=names[a], b=names[b], travel_time=round(seconds, 1)))

    return MapDefinition(
        name=name,
        width=params.width,
        height=params.height,
        bases=bases,
        edges=tuple(edges),
    )


def _production(params: MapGenParams, value: float) -> float:
    value = min(max(value, params.min_production), params.max_production)
    step = params.production_step
    # The second rounding removes float noise such as 1.2000000000000002.
    return round(round(value / step) * step, 6)


def _neutral_units(params: MapGenParams, value: float) -> int:
    return min(max(round(value), params.min_neutral_units), params.max_neutral_units)


def _starting_units(params: MapGenParams, value: float) -> int:
    return min(max(round(value), params.min_starting_units), params.max_starting_units)


def _route(a: int, b: int) -> Route:
    return (a, b) if a < b else (b, a)


def _routes_cross(positions: list[Point], first: Route, second: Route) -> bool:
    """True when two routes intersect somewhere other than a shared base."""

    if set(first) & set(second):
        return False
    a, b = positions[first[0]], positions[first[1]]
    c, d = positions[second[0]], positions[second[1]]
    return _side(c, d, a) * _side(c, d, b) < 0 and _side(a, b, c) * _side(a, b, d) < 0


def _side(a: Point, b: Point, point: Point) -> int:
    """Positive on one side of the line a-b, negative on the other, zero on it."""

    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def _route_is_clear(positions: list[Point], route: Route, clearance: float) -> bool:
    return all(
        _distance_to_route(positions, index, route) >= clearance
        for index in range(len(positions))
        if index not in route
    )


def _distance_to_route(positions: list[Point], index: int, route: Route) -> float:
    point = positions[index]
    a, b = positions[route[0]], positions[route[1]]
    dx, dy = b[0] - a[0], b[1] - a[1]
    along = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy)
    along = min(max(along, 0.0), 1.0)
    return math.dist(point, (a[0] + along * dx, a[1] + along * dy))


def _connected(count: int, routes: set[Route]) -> bool:
    neighbours: dict[int, list[int]] = {index: [] for index in range(count)}
    for a, b in routes:
        neighbours[a].append(b)
        neighbours[b].append(a)
    visited = {0}
    stack = [0]
    while stack:
        for neighbour in neighbours[stack.pop()]:
            if neighbour not in visited:
                visited.add(neighbour)
                stack.append(neighbour)
    return len(visited) == count
