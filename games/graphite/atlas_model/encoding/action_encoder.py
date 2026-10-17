"""Turn the network's choice into a move, and say which choices are legal.

The game accepts any route and any whole number of units, which is far too
many possibilities to learn one by one. The network instead makes two small
choices:

1. wait, or pick a route: a source base and a connected target base
2. pick how much to send, from the short list in ``AMOUNT_OPTIONS``

``legal_actions`` lists every choice the game would accept in a position,
together with the number of units each one sends. The network is only ever
allowed to pick from that list, so it never makes an illegal move.
"""

from dataclasses import dataclass
from math import ceil, floor
from typing import NamedTuple

import numpy as np

from game_types import NEUTRAL, BaseView, GameView, SendAction

from atlas_model.encoding.forecast import BaseTimeline, build_timelines, ticks_until_arrival
from atlas_model.encoding.state_encoder import MapLayout, layout_for


# ---- Amount options ------------------------------------------------------------
# Shares of the whole units standing at the source base, rounded up. Eighths
# reproduce 93% of strategic_ai's sends exactly; quarters manage only 54%.
SEND_FRACTIONS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
# "just_enough" is the smallest fleet that takes the target when it lands,
# counting every fleet already in flight. It is offered only when the source
# can afford it and the target would not be mine anyway.
AMOUNT_OPTIONS = ("just_enough", *(f"{fraction * 100:g}%" for fraction in SEND_FRACTIONS))
JUST_ENOUGH = AMOUNT_OPTIONS.index("just_enough")


class SendChoice(NamedTuple):
    """What the network picks when it does not wait. Waiting is ``None``."""

    source: int  # row of the source base
    target: int  # row of the target base
    amount: int  # position in AMOUNT_OPTIONS


@dataclass(frozen=True)
class LegalActions:
    """Every send the game would accept right now. Waiting is always allowed."""

    base_ids: tuple[str, ...]
    allowed: np.ndarray  # [bases, bases, amount options], True where legal
    units: np.ndarray  # [bases, bases, amount options], units that choice sends


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
    amounts: list[list[int]] = []
    for source, base in base_at.items():
        # The game only lets whole units leave, so the fraction is dropped.
        available = int(base.units)
        if base.owner != me or available < 1:
            continue
        shares = [max(1, ceil(fraction * available - 1e-9)) for fraction in SEND_FRACTIONS]

        for target, travel_seconds in layout.neighbours[source]:
            target_base = base_at[target]
            needed = _units_to_capture(
                target_base,
                timelines.get(target_base.id),
                ticks_until_arrival(travel_seconds, seconds_per_tick),
                me,
                seconds_per_tick,
            )
            sources.append(source)
            targets.append(target)
            # Zero marks "just enough" as not on offer for this route.
            amounts.append([needed if 1 <= needed <= available else 0, *shares])

    units = np.zeros((count, count, len(AMOUNT_OPTIONS)), dtype=np.int32)
    if sources:
        units[sources, targets] = amounts
    return LegalActions(base_ids=layout.base_ids, allowed=units > 0, units=units)


def decode_action(choice: SendChoice | None, legal: LegalActions) -> SendAction | None:
    """The network's choice as a move the game understands."""

    if choice is None:
        return None
    if not legal.allowed[choice]:
        raise ValueError(f"{choice} is not a legal choice in this position")
    return SendAction(
        source_id=legal.base_ids[choice.source],
        target_id=legal.base_ids[choice.target],
        units=int(legal.units[choice]),
    )


def encode_action(action: SendAction | None, legal: LegalActions) -> SendChoice | None:
    """The choice closest to a move someone else made, for learning from a teacher.

    The route is kept and the amount becomes the nearest option. A move the
    game would reject does nothing, so it is recorded as waiting.
    """

    if action is None:
        return None
    try:
        source = legal.base_ids.index(action.source_id)
        target = legal.base_ids.index(action.target_id)
    except ValueError:
        return None
    options = legal.allowed[source, target]
    if not options.any() or not 1 <= action.units <= legal.units[source, target].max():
        return None

    distance = np.abs(legal.units[source, target] - action.units)
    distance[~options] = np.iinfo(np.int32).max
    return SendChoice(source, target, int(distance.argmin()))


def stack_legal_actions(legal: list[LegalActions]) -> np.ndarray:
    """Masks from maps of different sizes as one padded batch; padding is never allowed."""

    largest = max(len(entry.base_ids) for entry in legal)
    batch = np.zeros((len(legal), largest, largest, len(AMOUNT_OPTIONS)), dtype=bool)
    for row, entry in enumerate(legal):
        count = len(entry.base_ids)
        batch[row, :count, :count] = entry.allowed
    return batch


def _units_to_capture(
    target: BaseView,
    timeline: BaseTimeline | None,
    tick: int,
    me: int,
    seconds_per_tick: float,
) -> int:
    """The smallest fleet, sent now and landing on ``tick``, that leaves the target mine.

    Zero or less means the target will already be mine at that moment.
    """

    if timeline is None:
        # No fleets on their way: only the garrison, grown until the landing, stands there.
        holder = target.owner
        garrison = target.units
        if holder != NEUTRAL:
            garrison += target.production_rate * seconds_per_tick * tick
        forces = {holder: garrison} if garrison > 1e-9 else {}
    else:
        holder, forces = timeline.forces_on(tick)

    mine = forces.pop(me, 0.0)
    if not forces:
        # Nobody to beat. An empty base still needs one unit to claim it.
        return 0 if holder == me or mine > 0 else 1

    # Winning needs strictly more than the strongest rival. The small margin
    # guards against a rival total that is a hair above a whole number.
    return floor(max(forces.values()) - mine + 1e-6) + 1
