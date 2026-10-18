"""Atlas Clean: Atlas's look-ahead search, without a copy of any opponent.

Atlas (``ais/atlas_ai.py``) forecasts its opponents with exact copies of them:
the code of the three scripted AIs, and the neural network's own checkpoint.
This version forecasts every opponent with Atlas's own ten policy styles only.
It watches the fleets each opponent launches and, per opponent, picks the
style of its own that best explains them, so it still adapts to what it sees
on the board. When no style fits well, it also plans against a cautious
opponent and scores every plan on its average and its worst outcome.

Everything else is Atlas's: the exact simulator (``_atlas_sim``), the tactical
policies (``_atlas_policy``), and the search over its policy styles, replanned
every game second, coarsely 30 seconds ahead and then tick by tick for the
three best styles.

Left out with the copies: the "certified" winning lines, which Atlas keeps only
while an opponent matches one of the copied scripted AIs exactly.

Planning is spread over the ticks. A full plan simulates about 2,500 steps
(twice that when it also plans against the cautious opponent), which takes
up to a few seconds on a large map. Done in one go, as Atlas does, that
freezes the game window once per game second. Here every decision does at
most ``WORK_PER_TICK`` simulation steps of the current plan and keeps playing
the style it already chose; a plan therefore arrives about two to five
game seconds after the position it was made for (measured: a new plan every
2.5 to 4 game seconds). The budget counts steps, not
time, so a match always plays out the same way.

Memory: a player remembers what it has seen during one match. ``new_player()``
gives a fresh player, so any number of matches can run side by side in one
process. The module's ``choose_action`` is one shared player for the game's
own window, which plays one match at a time; it keeps the seats apart, so it
can also play several seats of that match.
"""

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from functools import partial
from math import ceil

from game_types import GameView, SendAction

from ais._atlas_policy import _position, policy as base_policy
from ais._atlas_sim import Simulation


# Atlas's ten policy styles, from aggressive (low reserves, full commitment) to
# patient (reserves, saving up for valuable targets).
STYLES = [
    dict(reserve=0.1, buffer=0.0, saving=False, holding=0.0, costscale=20.0, surge=1.0, rally=1.0, overwhelm=2.0, enemy=3.0),
    dict(reserve=0.0, buffer=0.0, saving=False, holding=0.0, costscale=20.0, surge=1.0, rally=1.0, enemy=3.0),
    dict(reserve=0.0, buffer=0.0, saving=False, holding=0.0, costscale=20.0, surge=1.0, enemy=3.0),
    dict(reserve=0.5, buffer=0.0, saving=False, holding=0.0, costscale=10.0, surge=0.9),
    dict(reserve=1.0, buffer=0.0, saving=False, holding=0.0, costscale=10.0, surge=0.6),
    dict(reserve=1.0, buffer=0.0, saving=False, holding=0.0, costscale=5.0),
    dict(reserve=1.0, buffer=0.0, saving=False, holding=0.65, costscale=5.0),
    dict(reserve=0.5, buffer=0.0, saving=False, holding=0.25, costscale=20.0),
    dict(reserve=1.0, buffer=0.0, saving=True, holding=0.25, costscale=10.0, horizon=40.0),
    dict(reserve=2.0, buffer=2.0, saving=True, holding=0.5, costscale=20.0),
]
# The opponent assumed when none of the styles explains its moves.
CAUTIOUS = dict(reserve=1.0, holding=0.3, buffer=0.0, saving=False, surge=0.6)

FINE_HORIZON = 12.0  # seconds simulated tick by tick in the coarse search
SEARCH_HORIZON = 30.0  # seconds simulated in every search
REPLAN_SECONDS = 1.0  # a new plan starts at most this often, and only once the last one is done
WORK_PER_TICK = 120  # simulation steps per decision; measured 35-50 ms per decision on 8-24 bases, under load
REFINED_STYLES = 3  # the best styles of the coarse search, searched again tick by tick
NO_FIT = 6.0  # above this, the cautious opponent is planned against as well
ERROR_MEMORY = 0.97  # each tick, older prediction errors count this much less
MAX_CONTEXTS = 12


def policy(state: GameView, **style) -> SendAction | None:
    """Atlas's policy, plus its guard: a contested neutral that out-produces the source gets a garrison."""

    action = base_policy(state, **style)
    if action is None:
        return None
    position = _position(state)
    i, j = position.index[action.source_id], position.index[action.target_id]
    source, target = position.bases[i], position.bases[j]
    contested = any(position.bases[k].owner not in (0, state.player_id) for k, _ in position.graph[j])
    if target.owner == 0 and target.production_rate > source.production_rate and contested:
        travel = next(t for k, t in position.graph[i] if k == j)
        need = position.required(j, travel)
        available = position.available(i)
        if need and available >= action.units:
            amount = min(available, max(action.units, need + ceil(target.production_rate)))
            return SendAction(action.source_id, action.target_id, amount)
    return action


STYLE_POLICIES = [partial(policy, **style) for style in STYLES]


def action_distance(predicted: SendAction | None, actual: SendAction | None) -> float:
    """How wrong a prediction was: 0 if exact, 1 for another move, in between for another army size."""

    if predicted == actual:
        return 0.0
    if predicted is None or actual is None:
        return 1.0
    if (predicted.source_id, predicted.target_id) != (actual.source_id, actual.target_id):
        return 1.0
    return min(1.0, abs(predicted.units - actual.units) / max(1, predicted.units, actual.units))


@dataclass
class _Match:
    """What one seat remembers about one match."""

    previous: GameView
    action: SendAction | None = None
    style: dict = field(default_factory=lambda: STYLES[0])
    next_plan: int = 0
    planner: object = None  # the plan in progress, a generator; None between plans
    plans_done: int = 0
    # Per opponent, each style's discounted error in predicting its launches.
    errors: dict[int, list[float]] = field(default_factory=dict)


class AtlasClean:
    """One player with its own memory. Call it with each new view, as the game calls ``choose_action``.

    ``work_per_tick=None`` plans all at once, as Atlas does (for comparisons).
    """

    def __init__(self, work_per_tick: int | None = WORK_PER_TICK):
        self._matches: OrderedDict = OrderedDict()
        self.work_per_tick = work_per_tick

    def __call__(self, state: GameView) -> SendAction | None:
        # A view has no match number, so a match is recognised by seat, map, and an unbroken run of ticks.
        key = (state.player_id, state.seconds_per_tick, tuple((b.id, b.production_rate) for b in state.bases), state.edges)
        match = self._matches.get(key)
        if match is not None and match.previous == state:
            return match.action
        if match is None or state.tick != match.previous.tick + 1:
            match = _Match(previous=state, next_plan=state.tick)
            self._matches[key] = match
            while len(self._matches) > MAX_CONTEXTS:
                self._matches.popitem(last=False)
            if state.tick == 0 and len({b.owner for b in state.bases} - {0}) > 2:
                # With two opponents, spend one tick watching before committing an army.
                match.action = None
                return None
        else:
            self._learn_from_launches(match, state)

        if match.planner is None and state.tick >= match.next_plan:
            match.planner = self._search(match, state)
            match.next_plan = state.tick + max(1, round(REPLAN_SECONDS / state.seconds_per_tick))
        if match.planner is not None:
            self._work_on_plan(match)
        match.action = policy(state, **match.style)
        match.previous = state
        return match.action

    def _learn_from_launches(self, match: _Match, state: GameView) -> None:
        """Score every style on how well it would have predicted each opponent's last move."""

        previous = match.previous
        if min((edge.travel_time for edge in state.edges), default=0) <= state.seconds_per_tick:
            # A fleet could launch and land between two views, so a missing launch proves nothing.
            return
        for enemy in {b.owner for b in previous.bases} - {0, state.player_id}:
            launched = [
                fleet for fleet in state.fleets
                if fleet.owner == enemy and abs(fleet.total_time - fleet.remaining_time - state.seconds_per_tick) < 1e-07
            ]
            actual = SendAction(launched[0].source_id, launched[0].target_id, launched[0].units) if launched else None
            seen_by_enemy = replace(previous, player_id=enemy)
            predictions = [style(seen_by_enemy) for style in STYLE_POLICIES]
            if actual is not None or any(prediction is not None for prediction in predictions):
                errors = match.errors.setdefault(enemy, [0.0] * len(STYLES))
                for i, prediction in enumerate(predictions):
                    errors[i] = errors[i] * ERROR_MEMORY + action_distance(prediction, actual)

    def _work_on_plan(self, match: _Match) -> None:
        """Advance the plan in progress by this decision's share of simulation steps."""

        budget = self.work_per_tick
        while budget is None or budget > 0:
            try:
                next(match.planner)
            except StopIteration as finished:
                match.style = finished.value
                match.planner = None
                match.plans_done += 1
                return
            if budget is not None:
                budget -= 1

    def _search(self, match: _Match, state: GameView):
        """The style whose simulated future looks best against the forecast opponents.

        A generator: it pauses after every simulation step and returns the style when done.
        """

        me = state.player_id
        players = sorted({b.owner for b in state.bases} - {0} | {f.owner for f in state.fleets})
        opponents = [p for p in players if p != me]
        no_errors = [0.0] * len(STYLES)
        best_fit = {p: min(range(len(STYLES)), key=match.errors.get(p, no_errors).__getitem__) for p in opponents}
        scenarios = [{p: STYLE_POLICIES[best_fit[p]] for p in opponents}]
        if any(min(match.errors.get(p, no_errors)) > NO_FIT for p in opponents):
            cautious = dict(scenarios[0])
            for p in opponents:
                if min(match.errors.get(p, no_errors)) > NO_FIT:
                    cautious[p] = partial(policy, **CAUTIOUS)
            scenarios.append(cautious)

        dt = state.seconds_per_tick
        coarse = max(1, round(0.5 / dt))
        coarse_steps = (
            [1] * max(1, round(FINE_HORIZON / dt))
            + [coarse] * max(1, round((SEARCH_HORIZON - FINE_HORIZON) / (coarse * dt)))
        )
        ranked = []
        for index, style in enumerate(STYLES):
            score = yield from self._score(state, style, scenarios, coarse_steps)
            ranked.append((score, index))
        ranked.sort(reverse=True)
        # Atlas refines whenever an opponent is not one of its copies, which is now always.
        fine_steps = [1] * max(1, round(SEARCH_HORIZON / dt))
        best_score, best_style = -float("inf"), match.style
        for _, index in ranked[:REFINED_STYLES]:
            score = yield from self._score(state, STYLES[index], scenarios, fine_steps)
            if score > best_score:
                best_score, best_style = score, STYLES[index]
        return best_style

    @staticmethod
    def _score(state: GameView, style: dict, scenarios: list[dict], steps: list[int]) -> float:
        """Play the style against each scenario; 65% of the average outcome, 35% of the worst.

        A generator, pausing after every simulation step.
        """

        me = state.player_id
        horizon = sum(steps) * state.seconds_per_tick
        outcomes = []
        for opponents in scenarios:
            simulation = Simulation(state)
            players = [me, *opponents]
            running_total = 0.0
            for duration in steps:
                actions = {
                    p: policy(simulation.view(p), **style) if p == me else opponents[p](simulation.view(p))
                    for p in players
                }
                simulation.step(actions, duration)
                yield
                value = simulation.value(me)
                if abs(value) > 100000.0:
                    # Someone was eliminated: the sooner, the more it counts.
                    value += (1 if value > 0 else -1) * (horizon - simulation.tick * state.seconds_per_tick) * 100
                    break
                running_total += value / len(steps)
            outcomes.append(value + 0.1 * running_total)
        return 0.65 * sum(outcomes) / len(outcomes) + 0.35 * min(outcomes)


def new_player() -> AtlasClean:
    """A player with fresh memory, for one match. Runners that host several matches at once use this."""

    return AtlasClean()


_shared = AtlasClean()


def choose_action(state: GameView) -> SendAction | None:
    return _shared(state)
