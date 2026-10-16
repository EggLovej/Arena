"""Turn a GameView into numbers a network can read.

The result is three tables, all from the acting player's point of view:

- base features: one row per base; changes every tick
- pair features: one cell per ordered pair of bases; fixed for the whole match
- global features: a few numbers about the position as a whole

Three rules shape the encoding:

1. "Mine" and "enemy" replace Red and Blue, so both seats look the same.
2. Base names, list order, and screen coordinates are left out, so nothing
   about one particular map can be memorised.
3. Counts are squashed with a logarithm, so the step from 5 to 10 units is as
   visible as the step from 50 to 100.

1v1 only for now: any owner other than the acting player or neutral is "the
enemy".
"""

from dataclasses import dataclass
from math import exp, log1p

import numpy as np

from game_types import NEUTRAL, GameView

from atlas_model.encoding.forecast import BaseTimeline, build_timelines


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
)
PAIR_FEATURES = (
    "connected",  # 1 if a route joins the two bases
    "travel_time",  # seconds along that route, 0 if not connected
    "shortest_path_time",  # seconds along the fastest chain of routes
)
GLOBAL_FEATURES = (
    "my_units_share",  # my units out of mine plus the enemy's, fleets included
    "my_production_share",  # my production out of mine plus the enemy's
    "neutral_production_share",  # production still unclaimed, out of all production
    "time",  # how far into the match we are
)


@dataclass(frozen=True)
class MapLayout:
    """The parts of a position that never change during a match."""

    base_ids: tuple[str, ...]
    index: dict[str, int]  # base ID -> row number
    # For each row, the rows it has a route to and the travel time in seconds.
    neighbours: tuple[tuple[tuple[int, float], ...], ...]
    pair_features: np.ndarray  # [bases, bases, pair features]


@dataclass(frozen=True)
class EncodedState:
    base_ids: tuple[str, ...]  # row number -> base ID, needed to turn a choice back into a move
    base_features: np.ndarray  # [bases, base features]
    pair_features: np.ndarray  # [bases, bases, pair features]
    global_features: np.ndarray  # [global features]


@dataclass(frozen=True)
class StateBatch:
    """Several positions stacked, padded to the largest map among them."""

    base_features: np.ndarray  # [batch, bases, base features]
    pair_features: np.ndarray  # [batch, bases, bases, pair features]
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
    return MapLayout(
        base_ids=base_ids,
        index=index,
        neighbours=tuple(tuple(sorted(routes)) for routes in neighbours),
        pair_features=pair_features,
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
    for fleet in state.fleets:
        all_units += fleet.units
        if fleet.owner == me:
            my_units += fleet.units
        else:
            enemy_units += fleet.units

    # Rows are collected as plain lists and converted once; that is much
    # faster than writing into the array one base at a time.
    rows: list = [None] * len(layout.base_ids)
    for base in state.bases:
        owner = base.owner
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

        rows[layout.index[base.id]] = (
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
    base_features = np.array(rows, dtype=np.float32)

    all_production = my_production + enemy_production + neutral_production
    global_features = np.array(
        (
            _share(my_units, enemy_units),
            _share(my_production, enemy_production),
            neutral_production / all_production if all_production > 0 else 0.0,
            state.time_seconds / MATCH_SECONDS_REFERENCE,
        ),
        dtype=np.float32,
    )

    return EncodedState(
        base_ids=layout.base_ids,
        base_features=base_features,
        pair_features=layout.pair_features,
        global_features=global_features,
    )


def stack_states(states: list[EncodedState]) -> StateBatch:
    """Combine positions from maps of different sizes into one padded batch."""

    largest = max(len(state.base_ids) for state in states)
    batch = len(states)
    base_features = np.zeros((batch, largest, len(BASE_FEATURES)), dtype=np.float32)
    pair_features = np.zeros((batch, largest, largest, len(PAIR_FEATURES)), dtype=np.float32)
    global_features = np.zeros((batch, len(GLOBAL_FEATURES)), dtype=np.float32)
    base_mask = np.zeros((batch, largest), dtype=bool)

    for row, state in enumerate(states):
        count = len(state.base_ids)
        base_features[row, :count] = state.base_features
        pair_features[row, :count, :count] = state.pair_features
        global_features[row] = state.global_features
        base_mask[row, :count] = True
    return StateBatch(base_features, pair_features, global_features, base_mask)


# A count is squashed as log(1 + count) and divided by the same for its reference.
_UNITS_SCALE = 1 / log1p(UNITS_REFERENCE)
_PRODUCTION_SCALE = 1 / log1p(PRODUCTION_REFERENCE)


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
