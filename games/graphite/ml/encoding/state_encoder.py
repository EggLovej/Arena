"""Turn a GameView into numbers a network can read.

The result is four tables, all from the acting player's point of view:

- base features: one row per base; changes every tick
- pair features: one cell per ordered pair of bases; fixed for the whole match
- route features: one cell per route and direction; changes every tick
- global features: a few numbers about the position as a whole

Three rules shape the encoding:

1. "Mine" and "enemy" replace Red and Blue, so both seats look the same.
2. Base names, list order, and screen coordinates are left out, so nothing
   about one particular map can be memorised.
3. Counts are squashed with a logarithm, so the step from 5 to 10 units is as
   visible as the step from 50 to 100. Since 2026-10-08 every count is also
   there as a plain number (divided by a reference, so it stays near 0 to a
   few), for networks that read the "v2" set (see ``FEATURE_SETS``).

1v1 only for now: any owner other than the acting player or neutral is "the
enemy". The exception is the production shares at the end of the global
features, which already have a place for a second enemy (0 in a 1v1).

Features are only ever added at the end of a list, never changed or removed
in place. A model reads the first N of each list, as many as it was trained
with, so older models keep working when new features arrive.
"""

from dataclasses import dataclass
from math import exp, log1p

import numpy as np

from game_types import NEUTRAL, GameView

from ml.encoding.forecast import BaseTimeline, build_timelines, ticks_until_arrival, units_to_capture


# ---- Scales ------------------------------------------------------------------
# Networks learn best from inputs of roughly the same size, around 0 to 1.
# Each reference is the value that maps to 1.0; larger values are still allowed.
UNITS_REFERENCE = 250.0
PRODUCTION_REFERENCE = 10.0
TRAVEL_SECONDS_REFERENCE = 5.0  # one route
PATH_SECONDS_REFERENCE = 15.0  # a journey over several routes
MATCH_SECONDS_REFERENCE = 180.0
# "Arrives soon" is 1.0 for a fleet landing now and fades with this time constant.
ARRIVAL_FADE_SECONDS = 5.0
# How long a capture takes to earn back its cost; longer is cut off here.
PAYBACK_SECONDS_REFERENCE = 60.0
PAYBACK_SECONDS_LIMIT = 600.0
# A capture costing more than this many times the source's units is shown as this many.
COST_SHARE_LIMIT = 4.0
# Plain (not logarithmic) counts are divided by this.
LINEAR_UNITS_REFERENCE = 100.0
# The forecast "if nobody sends anything new" is shown this many ticks ahead (0.5 to 8 seconds).
HORIZON_TICKS = (5, 10, 20, 50, 80)

# ---- What each column means ----------------------------------------------------
BASE_FEATURES = (
    "is_mine",
    "is_enemy",
    "is_neutral",
    "units",  # log scale
    "production",  # log scale
    "units_share",  # this base's units as a share of all units in the match
    "incoming_mine",  # my units on their way here, log scale
    "incoming_enemy",  # enemy units on their way here, log scale
    "mine_arrives_soon",  # 1 = my next fleet lands now, 0 = none coming
    "enemy_arrives_soon",  # 1 = the enemy's next fleet lands now, 0 = none coming
    "future_is_mine",  # owner once every fleet now in flight has landed
    "future_is_enemy",
    "future_is_neutral",
    "future_units",  # units at that moment, log scale
    # Added 2026-10-08 for the "v2" set: plain counts, the forecast at fixed horizons, and the front.
    "units_linear",  # units / 100
    "production_linear",  # production / 10
    "incoming_mine_linear",  # / 100
    "incoming_enemy_linear",  # / 100
    # Owner and units HORIZON_TICKS from now if nobody sends anything new, counting production
    # and every fleet in flight that has landed by then: in_5_ticks_is_mine, ..., in_80_ticks_units.
    *(f"in_{ticks}_ticks_{part}" for ticks in HORIZON_TICKS for part in ("is_mine", "is_enemy", "is_neutral", "units")),
    "future_units_linear",  # units once every fleet in flight has landed, / 100
    "has_non_mine_neighbour",  # 1 if a route leads to a base that is not mine
    "distance_to_enemy",  # shortest travel time to the nearest enemy base, / 15 (0 for an enemy base)
)
PAIR_FEATURES = (
    "connected",  # 1 if a route joins the two bases
    "travel_time",  # seconds along that route, 0 if not connected
    "shortest_path_time",  # seconds along the fastest chain of routes
)
# For every route, in each direction, from the point of view of whoever owns
# the source base: what would it take to make the target theirs with a fleet
# sent now? This is the sum strategic_ai does before every attack or defence:
# garrison plus production until the fleet lands, plus or minus every fleet
# that lands by then. All three are 0 when the target will be theirs anyway,
# and on routes out of a neutral base.
ROUTE_FEATURES = (
    "capture_cost",  # units needed, log scale
    "capture_cost_share",  # units needed as a share of the units at the source; above 1 is out of reach
    "payback_time",  # seconds of the target's production to earn the cost back, log scale
    # Added 2026-10-08 for the "v2" set.
    "capture_cost_linear",  # units needed / 100
    # How much nearer the nearest enemy base of the source's owner the target is than the
    # source, in shortest travel time / 15: above 0 leads towards the front, below 0 away.
    # Also on routes where a capture costs nothing; 0 out of a neutral base.
    "forward",
)
GLOBAL_FEATURES = (
    "my_units_share",  # my units out of mine plus the enemy's, fleets included
    "my_production_share",  # my production out of mine plus the enemy's
    "neutral_production_share",  # production still unclaimed, out of all production
    "time",  # how far into the match we are
    # Everyone's production as a share of all production on the map; with
    # neutral_production_share these add up to 1. Enemies are ranked by
    # production, so the slots mean the same in every match; a missing or
    # eliminated enemy is 0, so a 1v1 is a three-player game whose third
    # player has nothing.
    "my_production_of_all",
    "strongest_enemy_production_of_all",
    "second_enemy_production_of_all",
)


# How many columns each list had before the 2026-10-08 additions. A network built without
# saying otherwise reads the first this many, as every network did before; the new columns are
# read by name, through a feature set.
COLUMNS_BEFORE_V2 = {
    "base_features": BASE_FEATURES.index("units_linear"),
    "route_features": ROUTE_FEATURES.index("capture_cost_linear"),
}

# Which columns of each table a network reads, by name. "v1" is what our networks up to
# 2026-10-08 read (the first 14, 3, 0 and 4 of the lists). "v2", chosen on 2026-10-08: counts as
# plain numbers, the forecast at five horizons and after every fleet has landed, where the front
# is, and what a capture along each route costs.
FEATURE_SETS = {
    "v1": {
        "base_features": BASE_FEATURES[:14],
        "pair_features": PAIR_FEATURES[:3],
        "route_features": (),
        "global_features": GLOBAL_FEATURES[:4],
    },
    "v2": {
        "base_features": (
            "is_mine",
            "is_enemy",
            "is_neutral",
            "units_linear",
            "production_linear",
            "units_share",
            "incoming_mine_linear",
            "incoming_enemy_linear",
            *(f"in_{ticks}_ticks_{part}" for ticks in HORIZON_TICKS for part in ("is_mine", "is_enemy", "is_neutral", "units")),
            "future_is_mine",
            "future_is_enemy",
            "future_is_neutral",
            "future_units_linear",
            "has_non_mine_neighbour",
            "distance_to_enemy",
        ),
        "pair_features": PAIR_FEATURES[:3],
        "route_features": ("capture_cost_linear", "forward"),
        "global_features": GLOBAL_FEATURES[:4],
    },
}


def feature_columns(feature_set: str) -> dict[str, tuple[int, ...]]:
    """Where each column of a feature set sits in its table, per table."""

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Sets: {', '.join(FEATURE_SETS)}")
    lists = {
        "base_features": BASE_FEATURES,
        "pair_features": PAIR_FEATURES,
        "route_features": ROUTE_FEATURES,
        "global_features": GLOBAL_FEATURES,
    }
    return {table: tuple(lists[table].index(name) for name in names) for table, names in FEATURE_SETS[feature_set].items()}


@dataclass(frozen=True)
class MapLayout:
    """The parts of a position that never change during a match."""

    base_ids: tuple[str, ...]
    index: dict[str, int]  # base ID -> row number
    # For each row, the rows it has a route to and the travel time in seconds.
    neighbours: tuple[tuple[tuple[int, float], ...], ...]
    pair_features: np.ndarray  # [bases, bases, pair features]
    # Every route in both directions as (source row, target row), in the order
    # the route features are worked out.
    routes: np.ndarray  # [routes, 2]


@dataclass(frozen=True)
class EncodedState:
    base_ids: tuple[str, ...]  # row number -> base ID, needed to turn a choice back into a move
    base_features: np.ndarray  # [bases, base features]
    pair_features: np.ndarray  # [bases, bases, pair features]
    route_features: np.ndarray  # [bases, bases, route features], 0 where there is no route
    global_features: np.ndarray  # [global features]


@dataclass(frozen=True)
class StateBatch:
    """Several positions stacked, padded to the largest map among them."""

    base_features: np.ndarray  # [batch, bases, base features]
    pair_features: np.ndarray  # [batch, bases, bases, pair features]
    route_features: np.ndarray  # [batch, bases, bases, route features]
    global_features: np.ndarray  # [batch, global features]
    base_mask: np.ndarray  # [batch, bases], True for real bases and False for padding


def build_layout(state: GameView) -> MapLayout:
    """Do the once-per-match work: row numbers, routes, and shortest paths."""

    base_ids = tuple(base.id for base in state.bases)
    index = {base_id: row for row, base_id in enumerate(base_ids)}
    count = len(base_ids)

    connected = np.zeros((count, count))
    travel = np.zeros((count, count))
    shortest = np.full((count, count), np.inf)
    np.fill_diagonal(shortest, 0.0)
    neighbours = [[] for _ in range(count)]
    for edge in state.edges:
        a, b = index[edge.a], index[edge.b]
        neighbours[a].append((b, edge.travel_time))
        neighbours[b].append((a, edge.travel_time))
        connected[a, b] = connected[b, a] = 1.0
        travel[a, b] = travel[b, a] = edge.travel_time
        shortest[a, b] = shortest[b, a] = edge.travel_time

    # Floyd-Warshall: allow each base in turn as a stop-over and keep any shortcut it gives.
    for stop in range(count):
        shortest = np.minimum(shortest, shortest[:, stop : stop + 1] + shortest[stop : stop + 1, :])

    # Rounding hides the last-digit differences that come from adding route
    # times in a different order, so the same map always encodes identically.
    pair_features = np.stack(
        (
            connected,
            travel / TRAVEL_SECONDS_REFERENCE,
            np.round(shortest, 6) / PATH_SECONDS_REFERENCE,
        ),
        axis=-1,
    ).astype(np.float32)
    pair_features.setflags(write=False)
    neighbours = tuple(tuple(sorted(routes)) for routes in neighbours)
    routes = np.array(
        [(source, target) for source, routes in enumerate(neighbours) for target, _ in routes],
        dtype=np.int64,
    ).reshape(-1, 2)
    routes.setflags(write=False)
    return MapLayout(
        base_ids=base_ids,
        index=index,
        neighbours=neighbours,
        pair_features=pair_features,
        routes=routes,
    )


def encode_state(
    state: GameView,
    layout: MapLayout | None = None,
    timelines: dict[str, BaseTimeline] | None = None,
) -> EncodedState:
    """Encode one position.

    Pass the match's layout and the position's timelines if they are already
    built, to skip building them again.
    """

    if layout is None:
        layout = layout_for(state)
    if timelines is None:
        timelines = build_timelines(state)
    me = state.player_id
    fade_per_tick = state.seconds_per_tick / ARRIVAL_FADE_SECONDS

    my_units = enemy_units = my_production = enemy_production = 0.0
    neutral_production = all_units = 0.0
    production_by_enemy: dict[int, float] = {}
    for base in state.bases:
        all_units += base.units
        if base.owner == me:
            my_units += base.units
            my_production += base.production_rate
        elif base.owner == NEUTRAL:
            neutral_production += base.production_rate
        else:
            enemy_units += base.units
            enemy_production += base.production_rate
            production_by_enemy[base.owner] = production_by_enemy.get(base.owner, 0.0) + base.production_rate
    for fleet in state.fleets:
        all_units += fleet.units
        if fleet.owner == me:
            my_units += fleet.units
        else:
            enemy_units += fleet.units

    owners = [NEUTRAL] * len(layout.base_ids)
    for base in state.bases:
        owners[layout.index[base.id]] = base.owner
    # Each player's distance to its nearest enemy base, per row: mine for the base features,
    # every owner's for the routes out of its bases.
    to_enemy = {player: _distance_to_enemies(owners, layout, player) for player in {me, *owners} - {NEUTRAL}}

    # Rows are collected as plain lists and converted once; that is much
    # faster than writing into the array one base at a time.
    rows: list = [None] * len(layout.base_ids)
    # For the plain-number columns, worked out for all bases at once afterwards: units,
    # production, incoming mine, incoming enemy, units once every fleet has landed.
    counts: list = [None] * len(layout.base_ids)
    for base in state.bases:
        owner = base.owner
        row = layout.index[base.id]
        timeline = timelines.get(base.id)
        if timeline is None:
            incoming_mine = incoming_enemy = mine_soon = enemy_soon = 0.0
            future_owner, future_units = owner, base.units
        else:
            incoming_mine = timeline.incoming_units.get(me, 0.0)
            incoming_enemy = sum(timeline.incoming_units.values()) - incoming_mine
            mine_soon = enemy_soon = 0.0
            for fleet_owner, tick in timeline.first_arrival.items():
                soon = exp(-tick * fade_per_tick)
                if fleet_owner == me:
                    mine_soon = soon
                elif soon > enemy_soon:
                    enemy_soon = soon
            future_owner, future_units, _ = timeline.at()
        counts[row] = (base.units, base.production_rate, incoming_mine, incoming_enemy, future_units)

        rows[row] = (
            owner == me,
            owner != me and owner != NEUTRAL,
            owner == NEUTRAL,
            log1p(base.units) * _UNITS_SCALE,
            log1p(base.production_rate) * _PRODUCTION_SCALE,
            base.units / all_units if all_units > 0 else 0.0,
            log1p(incoming_mine) * _UNITS_SCALE,
            log1p(incoming_enemy) * _UNITS_SCALE,
            mine_soon,
            enemy_soon,
            future_owner == me,
            future_owner != me and future_owner != NEUTRAL,
            future_owner == NEUTRAL,
            log1p(future_units) * _UNITS_SCALE,
        )
    base_features = np.concatenate(
        (
            np.array(rows, dtype=np.float32),
            _plain_columns(state, layout, timelines, owners, counts, to_enemy[me]),
        ),
        axis=1,
    )

    all_production = my_production + enemy_production + neutral_production
    of_all = 1 / all_production if all_production > 0 else 0.0
    strongest, second = (sorted(production_by_enemy.values(), reverse=True) + [0.0, 0.0])[:2]
    global_features = np.array(
        (
            _share(my_units, enemy_units),
            _share(my_production, enemy_production),
            neutral_production * of_all,
            state.time_seconds / MATCH_SECONDS_REFERENCE,
            my_production * of_all,
            strongest * of_all,
            second * of_all,
        ),
        dtype=np.float32,
    )

    return EncodedState(
        base_ids=layout.base_ids,
        base_features=base_features,
        pair_features=layout.pair_features,
        route_features=_route_features(state, layout, timelines, to_enemy),
        global_features=global_features,
    )


def _route_features(
    state: GameView,
    layout: MapLayout,
    timelines: dict[str, BaseTimeline],
    to_enemy: dict[int, list[float]],
) -> np.ndarray:
    """The ROUTE_FEATURES table: what a capture along each route would cost its source's owner.

    ``to_enemy`` has each owner's distance to its nearest enemy base, per row.
    """

    count = len(layout.base_ids)
    table = np.zeros((count, count, len(ROUTE_FEATURES)), dtype=np.float32)
    if not len(layout.routes):
        return table
    base_at = [None] * count
    for base in state.bases:
        base_at[layout.index[base.id]] = base

    values = []
    for source, routes in enumerate(layout.neighbours):
        owner = base_at[source].owner
        units = int(base_at[source].units)
        for target, travel_seconds in routes:
            target_base = base_at[target]
            forward = 0.0 if owner == NEUTRAL else to_enemy[owner][source] - to_enemy[owner][target]
            cost = 0
            if owner != NEUTRAL:
                cost = units_to_capture(
                    target_base,
                    timelines.get(target_base.id),
                    ticks_until_arrival(travel_seconds, state.seconds_per_tick),
                    owner,
                    state.seconds_per_tick,
                )
            if cost <= 0:
                values.append((0.0, 0.0, 0.0, 0.0, forward))
                continue
            production = target_base.production_rate
            payback = min(cost / production, PAYBACK_SECONDS_LIMIT) if production > 0 else PAYBACK_SECONDS_LIMIT
            values.append(
                (
                    log1p(cost) * _UNITS_SCALE,
                    min(cost / max(units, 1), COST_SHARE_LIMIT),
                    log1p(payback) * _PAYBACK_SCALE,
                    cost / LINEAR_UNITS_REFERENCE,
                    forward,
                )
            )
    table[layout.routes[:, 0], layout.routes[:, 1]] = values
    return table


def _plain_columns(
    state: GameView,
    layout: MapLayout,
    timelines: dict[str, BaseTimeline],
    owners: list[int],
    counts: list[tuple[float, ...]],
    to_enemy: list[float],
) -> np.ndarray:
    """The base columns added on 2026-10-08, from ``units_linear`` on, for every base at once.

    Written into one array column by column: on maps this small, numpy's cost per call
    matters more than the arithmetic.
    """

    me = state.player_id
    count, horizons = len(owners), len(HORIZON_TICKS)
    columns = np.empty((count, _PLAIN_COLUMNS), dtype=np.float32)
    counts = np.array(counts)  # units, production, incoming mine, incoming enemy, units once all has landed
    columns[:, :4] = counts[:, :4] * _PLAIN_SCALES
    # The forecast at each horizon: first as if nothing were on its way anywhere (the owner
    # stays, an owned base grows), then the few bases with fleets coming, from their timelines.
    owner = np.array(owners)
    growth = counts[:, 1] * state.seconds_per_tick * (owner != NEUTRAL)
    later_units = counts[:, :1] + growth[:, None] * _HORIZONS
    later_owner = np.repeat(owner[:, None], horizons, axis=1)
    for base_id, timeline in timelines.items():
        row = layout.index[base_id]
        for column, (owner_then, units_then) in enumerate(timeline.at_each(HORIZON_TICKS)):
            later_owner[row, column] = owner_then
            later_units[row, column] = units_then
    ahead = columns[:, 4 : 4 + 4 * horizons].reshape(count, horizons, 4)
    ahead[..., 0] = later_owner == me
    ahead[..., 2] = later_owner == NEUTRAL
    ahead[..., 1] = 1.0 - ahead[..., 0] - ahead[..., 2]
    ahead[..., 3] = later_units / LINEAR_UNITS_REFERENCE
    columns[:, -3] = counts[:, 4] / LINEAR_UNITS_REFERENCE
    columns[:, -2] = [any(owners[other] != me for other, _ in routes) for routes in layout.neighbours]
    columns[:, -1] = to_enemy
    return columns


def _distance_to_enemies(owners: list[int], layout: MapLayout, player: int) -> list[float]:
    """Per row: shortest travel time to the nearest base of an enemy of ``player``, / 15.

    0 at an enemy base. With no enemy base left (its fleets may still be flying), every row
    gets the map's longest shortest path: as far away as anything can be on this map.
    """

    paths = layout.pair_features[..., PAIR_FEATURES.index("shortest_path_time")]
    enemy = [row for row, owner in enumerate(owners) if owner != player and owner != NEUTRAL]
    # A plain list: the encoders look single values up in Python loops, which is slow on an array.
    if not enemy:
        return [float(paths.max())] * len(owners)
    return paths[:, enemy].min(axis=1).tolist()


def stack_states(states: list[EncodedState]) -> StateBatch:
    """Combine positions from maps of different sizes into one padded batch."""

    largest = max(len(state.base_ids) for state in states)
    batch = len(states)
    base_features = np.zeros((batch, largest, len(BASE_FEATURES)), dtype=np.float32)
    pair_features = np.zeros((batch, largest, largest, len(PAIR_FEATURES)), dtype=np.float32)
    route_features = np.zeros((batch, largest, largest, len(ROUTE_FEATURES)), dtype=np.float32)
    global_features = np.zeros((batch, len(GLOBAL_FEATURES)), dtype=np.float32)
    base_mask = np.zeros((batch, largest), dtype=bool)

    for row, state in enumerate(states):
        count = len(state.base_ids)
        base_features[row, :count] = state.base_features
        pair_features[row, :count, :count] = state.pair_features
        route_features[row, :count, :count] = state.route_features
        global_features[row] = state.global_features
        base_mask[row, :count] = True
    return StateBatch(base_features, pair_features, route_features, global_features, base_mask)


# A count is squashed as log(1 + count) and divided by the same for its reference.
_UNITS_SCALE = 1 / log1p(UNITS_REFERENCE)
# For the plain columns: their number, the divisors of the first four, and the horizons as an array.
_PLAIN_COLUMNS = len(BASE_FEATURES) - BASE_FEATURES.index("units_linear")
_PLAIN_SCALES = 1 / np.array((LINEAR_UNITS_REFERENCE, PRODUCTION_REFERENCE, LINEAR_UNITS_REFERENCE, LINEAR_UNITS_REFERENCE))
_HORIZONS = np.array(HORIZON_TICKS, dtype=np.float64)
_PRODUCTION_SCALE = 1 / log1p(PRODUCTION_REFERENCE)
_PAYBACK_SCALE = 1 / log1p(PAYBACK_SECONDS_REFERENCE)


def _share(mine: float, theirs: float) -> float:
    total = mine + theirs
    return mine / total if total > 0 else 0.5


# Callers that only have a GameView, such as an AI file, reuse layouts through this cache.
_layout_cache: dict[tuple, MapLayout] = {}


def layout_for(state: GameView) -> MapLayout:
    """The layout for this position's map, built once and then remembered."""

    key = (tuple(base.id for base in state.bases), state.edges)
    layout = _layout_cache.get(key)
    if layout is None:
        if len(_layout_cache) >= 64:
            _layout_cache.clear()
        layout = _layout_cache[key] = build_layout(state)
    return layout
