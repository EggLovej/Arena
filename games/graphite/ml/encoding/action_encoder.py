"""Turn the network's choice into a move, and say which choices are legal.

The game accepts any route and any whole number of units, which is far too
many possibilities to learn one by one. The network instead makes two small
choices:

1. wait, or pick a route: a source base and a connected target base
2. pick how much to send, from the short list in ``AMOUNT_OPTIONS``, or, for
   a network with free amounts (since 2026-10-08, ``ml/models/amounts.py``),
   any whole number from 1 to the units at the source

``legal_actions`` lists every choice the game would accept in a position,
together with the number of units each one sends. The network is only ever
allowed to pick from that list, so it never makes an illegal move.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from game_types import GameView, SendAction

from ml.encoding.forecast import (
    BaseTimeline,
    build_timelines,
    ticks_until_arrival,
    units_to_capture,
    units_to_hold,
    units_to_spare,
)
from ml.encoding.state_encoder import MapLayout, layout_for


# ---- Amount options ------------------------------------------------------------
# Shares of the whole units standing at the source base, rounded up. Eighths
# reproduce 93% of strategic_ai's sends exactly; quarters manage only 54%.
SEND_FRACTIONS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
# "just_enough" is the smallest fleet that takes the target when it lands,
# counting every fleet already in flight. It is offered only when the source
# can afford it and the target would not be mine anyway.
# Two tactical amounts, added 2026-10-03 (options are only ever appended, so older
# networks keep reading the ones they know; see ``PolicyValueModel.forward``):
# "hold" is the smallest fleet that takes the target and still holds it after
# every fleet already on its way there has landed; for one of my own bases
# that would fall, the reinforcement it needs. Offered where it differs from
# "just_enough" and the source can afford it.
# "surplus" is everything the source can send and still hold out against every
# fleet already on its way to it. Offered only when that is less than all of
# its units (otherwise it is the 100% option) and at least one.
AMOUNT_OPTIONS = ("just_enough", *(f"{fraction * 100:g}%" for fraction in SEND_FRACTIONS), "hold", "surplus")
JUST_ENOUGH = AMOUNT_OPTIONS.index("just_enough")
HOLD = AMOUNT_OPTIONS.index("hold")
SURPLUS = AMOUNT_OPTIONS.index("surplus")


class SendChoice(NamedTuple):
    """What the network picks when it does not wait. Waiting is ``None``."""

    source: int  # row of the source base
    target: int  # row of the target base
    amount: int  # position in AMOUNT_OPTIONS; for a network with free amounts, the units themselves


# Per route, the four numbers every amount option is worked out from (see ``option_units``):
# - most: the units at the source, the most it can send (every whole number from 1 up to it
#   is legal for a network with free amounts; 0 where the route cannot be used)
# - needed: the smallest fleet that captures the target, which may be more than the source
#   has (0 when the target will be mine anyway)
# - hold: the smallest fleet that takes the target and still holds it once every fleet
#   already on its way there has landed (0 when nothing is needed)
# - surplus: what the source can spare and still hold out, where that is from 1 to less
#   than everything (0 otherwise)
LIMITS = ("most", "needed", "hold", "surplus")


@dataclass(frozen=True)
class LegalActions:
    """Every send the game would accept right now. Waiting is always allowed."""

    base_ids: tuple[str, ...]
    allowed: np.ndarray  # [bases, bases, amount options], True where legal
    units: np.ndarray  # [bases, bases, amount options], units that choice sends
    limits: np.ndarray  # [bases, bases, LIMITS], whole units; all 0 where the route cannot be used

    @property
    def most(self) -> np.ndarray:  # [bases, bases]
        return self.limits[..., 0]

    @property
    def needed(self) -> np.ndarray:  # [bases, bases]
        return self.limits[..., 1]


def legal_actions(
    state: GameView,
    layout: MapLayout | None = None,
    timelines: dict[str, BaseTimeline] | None = None,
) -> LegalActions:
    """List the legal sends. Pass the layout and timelines if they are already built."""

    if layout is None:
        layout = layout_for(state)
    if timelines is None:
        timelines = build_timelines(state)
    me = state.player_id
    seconds_per_tick = state.seconds_per_tick
    count = len(layout.base_ids)
    base_at = {layout.index[base.id]: base for base in state.bases}

    # Every route gets one entry in these lists; the arrays are filled once at
    # the end, which is much faster than writing to them route by route.
    sources: list[int] = []
    targets: list[int] = []
    limits: list[tuple[int, int, int, int]] = []  # LIMITS, per route
    for source, base in base_at.items():
        # The game only lets whole units leave, so the fraction is dropped.
        available = int(base.units)
        if base.owner != me or available < 1:
            continue
        spare = units_to_spare(base, timelines.get(base.id), me)
        surplus = spare if 1 <= spare < available else 0

        for target, travel_seconds in layout.neighbours[source]:
            target_base = base_at[target]
            timeline = timelines.get(target_base.id)
            arrival = ticks_until_arrival(travel_seconds, seconds_per_tick)
            needed = units_to_capture(target_base, timeline, arrival, me, seconds_per_tick)
            # Without fleets on their way to the target, holding takes just as much as taking it.
            hold = needed if timeline is None else units_to_hold(target_base, timeline, arrival, me, seconds_per_tick, needed)
            sources.append(source)
            targets.append(target)
            limits.append((available, max(needed, 0), max(hold, 0), surplus))

    table = np.zeros((count, count, len(LIMITS)), dtype=np.int32)
    units = np.zeros((count, count, len(AMOUNT_OPTIONS)), dtype=np.int32)
    if sources:
        table[sources, targets] = limits
        units[sources, targets] = option_units(table[sources, targets])
    return LegalActions(base_ids=layout.base_ids, allowed=units > 0, units=units, limits=table)


def option_units(limits: np.ndarray) -> np.ndarray:
    """What each amount option sends, from routes' LIMITS: [..., 4] -> [..., amount options], 0 where not on offer.

    The one place the options are defined, so recorded games, which keep only the
    limits, can rebuild them exactly.
    """

    most, needed, hold, surplus = (limits[..., part].astype(np.int64) for part in range(len(LIMITS)))
    # Shares of the units at the source, rounded up: as in Python, ceil(fraction * most).
    shares = np.maximum(1, np.ceil(most[..., None] * np.array(SEND_FRACTIONS) - 1e-9)).astype(np.int64)
    # "hold" is offered where it differs from "just enough".
    units = np.concatenate(
        (
            np.where((needed >= 1) & (needed <= most), needed, 0)[..., None],
            shares,
            np.where((hold >= 1) & (hold <= most) & (hold != needed), hold, 0)[..., None],
            surplus[..., None],
        ),
        axis=-1,
    )
    return np.where(most[..., None] >= 1, units, 0).astype(np.int32)


def decode_action(choice: SendChoice | None, legal: LegalActions, free: bool = False) -> SendAction | None:
    """The network's choice as a move the game understands.

    ``free``: the choice comes from a network with free amounts, so its amount is the units.
    """

    if choice is None:
        return None
    if free:
        if not 1 <= choice.amount <= legal.most[choice.source, choice.target]:
            raise ValueError(f"{choice} is not a legal choice in this position")
        units = choice.amount
    else:
        if not legal.allowed[choice]:
            raise ValueError(f"{choice} is not a legal choice in this position")
        units = legal.units[choice]
    return SendAction(
        source_id=legal.base_ids[choice.source],
        target_id=legal.base_ids[choice.target],
        units=int(units),
    )


def encode_action(action: SendAction | None, legal: LegalActions, free: bool = False) -> SendChoice | None:
    """The choice closest to a move someone else made, for learning from a teacher.

    The route is kept and the amount becomes the nearest option, or, with
    ``free`` amounts, stays exactly as it was. A move the game would reject
    does nothing, so it is recorded as waiting.
    """

    if action is None:
        return None
    try:
        source = legal.base_ids.index(action.source_id)
        target = legal.base_ids.index(action.target_id)
    except ValueError:
        return None
    if free:
        return SendChoice(source, target, action.units) if 1 <= action.units <= legal.most[source, target] else None
    options = legal.allowed[source, target]
    if not options.any() or not 1 <= action.units <= legal.units[source, target].max():
        return None

    distance = np.abs(legal.units[source, target] - action.units)
    distance[~options] = np.iinfo(np.int32).max
    return SendChoice(source, target, int(distance.argmin()))


def stack_route_limits(legal: list[LegalActions]) -> tuple[np.ndarray, np.ndarray]:
    """``most`` and ``needed`` from maps of different sizes as padded batches, as floats."""

    largest = max(len(entry.base_ids) for entry in legal)
    limits = np.zeros((len(legal), largest, largest, 2), dtype=np.float32)
    for row, entry in enumerate(legal):
        count = len(entry.base_ids)
        limits[row, :count, :count] = entry.limits[..., :2]
    return limits[..., 0], limits[..., 1]


def stack_legal_actions(legal: list[LegalActions]) -> np.ndarray:
    """Masks from maps of different sizes as one padded batch; padding is never allowed."""

    largest = max(len(entry.base_ids) for entry in legal)
    batch = np.zeros((len(legal), largest, largest, len(AMOUNT_OPTIONS)), dtype=bool)
    for row, entry in enumerate(legal):
        count = len(entry.base_ids)
        batch[row, :count, :count] = entry.allowed
    return batch
