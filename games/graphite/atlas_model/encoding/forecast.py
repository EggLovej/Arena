"""Work out what will happen at a base from the fleets already on their way.

Fleets cannot be recalled, so once they are in flight the outcome at their
target is fixed unless someone sends more. This follows the same rules as
``Game.step``: each tick a base first produces, then everything arriving that
tick fights at once, and the largest force keeps its lead over the second.

A ``BaseTimeline`` does that working-out once for one base. Both encoders
then look answers up in it, for both players, as often as they need.
"""

from bisect import bisect_right
from collections.abc import Iterable
from math import ceil
from typing import NamedTuple

from game_types import NEUTRAL, BaseView, FleetView, GameView


class Forecast(NamedTuple):
    owner: int
    units: float
    ticks: int  # how many ticks from now this describes


def ticks_until_arrival(remaining_time: float, seconds_per_tick: float) -> int:
    """How many ticks from now a fleet lands."""

    return max(1, ceil((remaining_time - 1e-9) / seconds_per_tick))


class BaseTimeline:
    """One base's future, as far as the fleets now in flight decide it."""

    def __init__(self, base: BaseView, incoming: Iterable[FleetView], seconds_per_tick: float):
        self.growth = base.production_rate * seconds_per_tick
        # Fleets landing on the same tick fight together, so group them by tick and owner.
        self.arrivals: dict[int, dict[int, float]] = {}
        self.incoming_units: dict[int, float] = {}  # owner -> total units on their way here
        self.first_arrival: dict[int, int] = {}  # owner -> tick their first fleet lands
        for fleet in incoming:
            tick = ticks_until_arrival(fleet.remaining_time, seconds_per_tick)
            owner = fleet.owner
            forces = self.arrivals.setdefault(tick, {})
            forces[owner] = forces.get(owner, 0.0) + fleet.units
            self.incoming_units[owner] = self.incoming_units.get(owner, 0.0) + fleet.units
            if tick < self.first_arrival.get(owner, tick + 1):
                self.first_arrival[owner] = tick

        # The base's owner and units right after each fight. Entry 0 is the present.
        self.ticks = [0]
        self.owners = [base.owner]
        self.units = [base.units]
        owner, units, now = base.owner, base.units, 0
        for tick in sorted(self.arrivals):
            if owner != NEUTRAL:
                units += self.growth * (tick - now)
            now = tick
            owner, units = _fight(owner, units, self.arrivals[tick])
            self.ticks.append(tick)
            self.owners.append(owner)
            self.units.append(units)

    def at(self, ticks: int | None = None) -> Forecast:
        """Owner and units ``ticks`` from now; by default, once the last fleet has landed."""

        if ticks is None:
            return Forecast(self.owners[-1], self.units[-1], self.ticks[-1])
        latest = bisect_right(self.ticks, ticks) - 1
        owner, units = self.owners[latest], self.units[latest]
        if owner != NEUTRAL:
            units += self.growth * (ticks - self.ticks[latest])
        return Forecast(owner, units, ticks)

    def forces_on(self, tick: int) -> tuple[int, dict[int, float]]:
        """Who holds the base going into ``tick``, and every force that will fight there.

        The forces are the garrison, after producing that tick, plus each
        owner's fleets landing on exactly that tick. A fleet sent now along a
        route that takes ``tick`` ticks joins this fight.
        """

        holder, garrison, _ = self.at(tick - 1)
        if holder != NEUTRAL:
            garrison += self.growth
        forces = dict(self.arrivals.get(tick, ()))
        if garrison > 1e-9:
            forces[holder] = forces.get(holder, 0.0) + garrison
        return holder, forces


def build_timelines(state: GameView) -> dict[str, BaseTimeline]:
    """A timeline for every base that has a fleet on its way. Other bases have none."""

    incoming: dict[str, list[FleetView]] = {}
    for fleet in state.fleets:
        incoming.setdefault(fleet.target_id, []).append(fleet)
    if not incoming:
        return {}
    return {
        base.id: BaseTimeline(base, incoming[base.id], state.seconds_per_tick)
        for base in state.bases
        if base.id in incoming
    }


def forecast_base(
    base: BaseView,
    incoming: Iterable[FleetView],
    seconds_per_tick: float,
    ticks: int | None = None,
) -> Forecast:
    """The base's owner and units ``ticks`` from now, if nobody sends anything new.

    Leave ``ticks`` unset to get the moment the last incoming fleet has landed.
    """

    return BaseTimeline(base, incoming, seconds_per_tick).at(ticks)


def forces_on_arrival(
    base: BaseView,
    incoming: Iterable[FleetView],
    seconds_per_tick: float,
    tick: int,
) -> tuple[int, dict[int, float]]:
    """See ``BaseTimeline.forces_on``."""

    return BaseTimeline(base, incoming, seconds_per_tick).forces_on(tick)


def _fight(owner: int, units: float, arriving: dict[int, float]) -> tuple[int, float]:
    """Resolve one tick's arrivals against the garrison, as ``Game._resolve_arrivals`` does."""

    forces = dict(arriving)
    if units > 1e-9:
        forces[owner] = forces.get(owner, 0.0) + units
    ranked = sorted(forces.items(), key=lambda item: item[1], reverse=True)
    strongest_owner, strongest = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if abs(strongest - second) <= 1e-9:
        return NEUTRAL, 0.0
    return strongest_owner, strongest - second
