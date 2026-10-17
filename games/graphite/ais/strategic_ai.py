"""A map-independent AI that plans attacks, defense, and reinforcement routes."""

from dataclasses import dataclass
from heapq import heappop, heappush
from math import ceil

from game_types import NEUTRAL, BaseView, GameView, SendAction


MIN_GARRISON = 2
DEFENSE_TARGET = 4


@dataclass(frozen=True)
class _Plan:
    score: float
    source_id: str
    target_id: str
    units: int

    def action(self) -> SendAction:
        return SendAction(self.source_id, self.target_id, self.units)


def choose_action(state: GameView) -> SendAction | None:
    """Choose one action without relying on map names, coordinates, or base IDs."""

    bases = {base.id: base for base in state.bases}
    graph = _build_graph(state)

    defense = _best_defense(state, bases, graph)
    if defense is not None:
        return defense.action()

    attack = _best_attack(state, bases, graph)
    if attack is not None:
        return attack.action()

    reinforcement = _best_reinforcement(state, bases, graph)
    if reinforcement is not None:
        return reinforcement.action()

    return None


def _build_graph(state: GameView) -> dict[str, list[tuple[str, float]]]:
    graph = {base.id: [] for base in state.bases}
    for edge in state.edges:
        graph[edge.a].append((edge.b, edge.travel_time))
        graph[edge.b].append((edge.a, edge.travel_time))
    for neighbors in graph.values():
        neighbors.sort()
    return graph


def _best_defense(
    state: GameView,
    bases: dict[str, BaseView],
    graph: dict[str, list[tuple[str, float]]],
) -> _Plan | None:
    """Reinforce friendly bases that have hostile fleets on the way."""

    plans = []
    threatened_bases = {
        fleet.target_id
        for fleet in state.fleets
        if fleet.owner != state.player_id
        and bases[fleet.target_id].owner == state.player_id
    }

    for target_id in sorted(threatened_bases):
        enemy_arrivals = [
            fleet.remaining_time
            for fleet in state.fleets
            if fleet.target_id == target_id and fleet.owner != state.player_id
        ]
        first_attack = min(enemy_arrivals)

        for source_id, travel_time in graph[target_id]:
            source = bases[source_id]
            if source.owner != state.player_id:
                continue

            horizon = max(first_attack, travel_time)
            projected = _signed_strength(state, bases[target_id], horizon)
            needed = ceil(DEFENSE_TARGET - projected)
            sendable = _sendable_units(state, source, graph)
            if needed <= 0 or sendable < needed:
                continue

            amount = min(sendable, needed + 2)
            score = (
                1_000
                + bases[target_id].production_rate * 20
                - first_attack * 8
                - travel_time * 3
            )
            plans.append(_Plan(score, source_id, target_id, amount))

    return _choose_plan(plans)


def _best_pressure_attack(
    state: GameView,
    bases: dict[str, BaseView],
    graph: dict[str, list[tuple[str, float]]],
) -> _Plan | None:
    """Coordinate several adjacent bases when none can attack alone."""

    plans = []
    enemy_targets = sorted(
        (
            base
            for base in state.bases
            if base.owner not in (state.player_id, NEUTRAL)
        ),
        key=lambda base: base.id,
    )
    for target in enemy_targets:
        sources = []
        for source_id, travel_time in graph[target.id]:
            source = bases[source_id]
            if source.owner != state.player_id:
                continue
            sendable = _sendable_units(state, source, graph)
            if sendable > 0:
                sources.append((source, travel_time, sendable))

        committed = sum(
            fleet.units
            for fleet in state.fleets
            if fleet.target_id == target.id and fleet.owner == state.player_id
        )
        if len(sources) < 2 and committed == 0:
            continue
        if not sources:
            continue

        first_arrival = min(travel_time for _, travel_time, _ in sources)
        projected = _signed_strength(state, target, first_arrival)
        committed_by_then = sum(
            fleet.units
            for fleet in state.fleets
            if fleet.target_id == target.id
            and fleet.owner == state.player_id
            and fleet.remaining_time <= first_arrival
        )
        campaign_requirement = ceil(1.5 - projected) + committed_by_then
        campaign_strength = committed + sum(sendable for _, _, sendable in sources)
        if campaign_strength * 2 < campaign_requirement:
            continue

        for source, travel_time, sendable in sources:
            score = (
                target.production_rate * 12
                + len(graph[target.id])
                + sendable * 0.4
                - travel_time
            )
            plans.append(_Plan(score, source.id, target.id, sendable))

    return _choose_plan(plans)


def _best_attack(
    state: GameView,
    bases: dict[str, BaseView],
    graph: dict[str, list[tuple[str, float]]],
) -> _Plan | None:
    """Capture valuable neighboring bases with enough projected strength."""

    plans = []
    for source in sorted(state.my_bases(), key=lambda base: base.id):
        sendable = _sendable_units(state, source, graph)
        if sendable <= 0:
            continue

        for target_id, travel_time in graph[source.id]:
            target = bases[target_id]
            if target.owner == state.player_id:
                continue

            projected = _signed_strength(state, target, travel_time)
            if projected >= 1:
                # Friendly fleets already on the way should finish this capture.
                continue

            needed = ceil(1.5 - projected)
            if needed > sendable:
                continue

            holding_buffer = max(1, ceil(target.production_rate * travel_time * 0.35))
            amount = min(sendable, needed + holding_buffer)
            enemy_bonus = 16 if target.owner != NEUTRAL else 6
            expansion_bonus = sum(
                bases[neighbor_id].owner != state.player_id
                for neighbor_id, _ in graph[target_id]
            )
            score = (
                enemy_bonus
                + target.production_rate * 18
                + len(graph[target_id]) * 1.5
                + expansion_bonus * 2
                - needed * 0.35
                - travel_time
            )
            plans.append(_Plan(score, source.id, target_id, amount))

    decisive_attack = _choose_plan(plans)
    if decisive_attack is not None:
        return decisive_attack
    return _best_pressure_attack(state, bases, graph)


def _best_reinforcement(
    state: GameView,
    bases: dict[str, BaseView],
    graph: dict[str, list[tuple[str, float]]],
) -> _Plan | None:
    """Move surplus from enclosed bases toward the most useful frontier."""

    frontier_ids = {
        base.id
        for base in state.my_bases()
        if any(
            bases[neighbor_id].owner != state.player_id
            for neighbor_id, _ in graph[base.id]
        )
    }
    if not frontier_ids:
        return None

    plans = []
    for source in sorted(state.my_bases(), key=lambda base: base.id):
        if source.id in frontier_ids:
            continue

        sendable = _sendable_units(state, source, graph)
        if sendable < 3:
            continue

        routes = _friendly_routes(state, source.id, bases, graph)
        destinations = []
        for frontier_id in frontier_ids:
            if frontier_id == source.id or frontier_id not in routes:
                continue
            distance, first_hop = routes[frontier_id]
            value = _frontier_value(state, bases, graph, frontier_id)
            destinations.append((value - distance * 1.2, frontier_id, first_hop))

        if not destinations:
            continue

        route_score, _, first_hop = max(destinations)
        amount = max(1, ceil(sendable * 0.7))
        score = source.units * 0.5 + route_score
        plans.append(_Plan(score, source.id, first_hop, amount))

    return _choose_plan(plans)


def _signed_strength(state: GameView, target: BaseView, horizon: float) -> float:
    """Project forces at a base; positive values favor the acting player."""

    if target.owner == state.player_id:
        strength = target.units + target.production_rate * horizon
    elif target.owner == NEUTRAL:
        strength = -target.units
    else:
        strength = -(target.units + target.production_rate * horizon)

    for fleet in state.fleets:
        if fleet.target_id != target.id or fleet.remaining_time > horizon:
            continue
        if fleet.owner == state.player_id:
            strength += fleet.units
        else:
            strength -= fleet.units
    return strength


def _sendable_units(
    state: GameView,
    source: BaseView,
    graph: dict[str, list[tuple[str, float]]],
) -> int:
    reserve = MIN_GARRISON
    touches_frontier = any(
        state.get_base(neighbor_id).owner != state.player_id
        for neighbor_id, _ in graph[source.id]
    )
    if touches_frontier:
        reserve = max(reserve, ceil(source.units * 0.18))

    incoming_enemy = sum(
        fleet.units
        for fleet in state.fleets
        if fleet.target_id == source.id and fleet.owner != state.player_id
    )
    incoming_friendly = sum(
        fleet.units
        for fleet in state.fleets
        if fleet.target_id == source.id and fleet.owner == state.player_id
    )
    reserve = max(reserve, incoming_enemy - incoming_friendly + DEFENSE_TARGET)
    return max(0, int(source.units) - reserve)


def _friendly_routes(
    state: GameView,
    source_id: str,
    bases: dict[str, BaseView],
    graph: dict[str, list[tuple[str, float]]],
) -> dict[str, tuple[float, str]]:
    """Return shortest friendly-only distances and the first hop for each."""

    routes = {source_id: (0.0, source_id)}
    queue = [(0.0, source_id, source_id)]
    while queue:
        distance, base_id, first_hop = heappop(queue)
        if distance > routes[base_id][0]:
            continue

        for neighbor_id, travel_time in graph[base_id]:
            if bases[neighbor_id].owner != state.player_id:
                continue
            next_distance = distance + travel_time
            next_hop = neighbor_id if base_id == source_id else first_hop
            previous = routes.get(neighbor_id)
            candidate = (next_distance, next_hop)
            if previous is None or candidate < previous:
                routes[neighbor_id] = candidate
                heappush(queue, (next_distance, neighbor_id, next_hop))
    return routes


def _frontier_value(
    state: GameView,
    bases: dict[str, BaseView],
    graph: dict[str, list[tuple[str, float]]],
    frontier_id: str,
) -> float:
    values = []
    for target_id, travel_time in graph[frontier_id]:
        target = bases[target_id]
        if target.owner == state.player_id:
            continue
        projected = _signed_strength(state, target, travel_time)
        opposition = max(0.0, -projected)
        enemy_bonus = 10 if target.owner != NEUTRAL else 3
        values.append(
            enemy_bonus
            + target.production_rate * 16
            + len(graph[target_id])
            - opposition * 0.3
            - travel_time
        )
    return max(values, default=0.0)


def _choose_plan(plans: list[_Plan]) -> _Plan | None:
    if not plans:
        return None
    return max(
        plans,
        key=lambda plan: (
            plan.score,
            plan.units,
            plan.source_id,
            plan.target_id,
        ),
    )
