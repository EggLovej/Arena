from collections import defaultdict
from game_types import BaseView, FleetView, GameView
from ais._atlas_policy import arrival_ticks, resolve

class Simulation:
    """Private forward simulator; arrival rules and float production match Game.

Future decision spacing is chosen by the caller. Combat still occurs at the
original tick, even between decisions. Forecasts never mutate the real game."""

    def __init__(self, state):
        self.state = state
        self.index = {b.id: i for i, b in enumerate(state.bases)}
        self.owners = [b.owner for b in state.bases]
        self.units = [b.units for b in state.bases]
        self.tick = 0
        self._cached_tick = None
        self._cached_view = None
        self.edges = {frozenset((e.a, e.b)): e.travel_time for e in state.edges}
        self.fleets = []
        for f in state.fleets:
            self.fleets.append((arrival_ticks(f.remaining_time, state.seconds_per_tick), f))
        self.players = tuple(sorted(set(self.owners) - {0} | {f.owner for f in state.fleets}))

    def view(self, player):
        s = self.state
        if self._cached_tick != self.tick:
            self._cached_view = GameView(player, s.tick + self.tick, (s.tick + self.tick) * s.seconds_per_tick, s.seconds_per_tick, tuple((BaseView(b.id, o, u, b.production_rate, b.x, b.y) for b, o, u in zip(s.bases, self.owners, self.units))), s.edges, tuple((FleetView(f.owner, f.units, f.source_id, f.target_id, f.remaining_time, f.total_time) for t, f in self.fleets)))
            self._cached_tick = self.tick
        v = self._cached_view
        return GameView(player, v.tick, v.time_seconds, v.seconds_per_tick, v.bases, v.edges, v.fleets)

    def step(self, actions, step_ticks):
        """Apply simultaneous sends, then advance original ticks through every arrival."""
        dt = self.state.seconds_per_tick
        for player, a in actions.items():
            if a is None:
                continue
            i = self.index[a.source_id]
            if self.owners[i] != player or a.units > int(self.units[i]):
                raise ValueError('bad simulation action')
            self.units[i] -= a.units
            travel = self.edges[frozenset((a.source_id, a.target_id))]
            f = FleetView(player, a.units, a.source_id, a.target_id, travel, travel)
            self.fleets.append((self.tick + arrival_ticks(travel, dt), f))
        end = self.tick + step_ticks
        events = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        remaining = []
        for t, f in self.fleets:
            if t <= end:
                events[t][self.index[f.target_id]][f.owner] += f.units
            else:
                remaining_time = f.remaining_time
                for _ in range(step_ticks):
                    remaining_time -= dt
                remaining.append((t, FleetView(f.owner, f.units, f.source_id, f.target_id, remaining_time, f.total_time)))
        self.fleets = remaining
        last = self.tick
        for t, arrivals in sorted(events.items()):
            for i, b in enumerate(self.state.bases):
                if self.owners[i]:
                    increment = b.production_rate * dt
                    for _ in range(t - last):
                        self.units[i] += increment
            for i, forces in arrivals.items():
                self.owners[i], self.units[i] = resolve(self.owners[i], self.units[i], forces)
            last = t
        for i, b in enumerate(self.state.bases):
            if self.owners[i]:
                increment = b.production_rate * dt
                for _ in range(end - last):
                    self.units[i] += increment
        self.tick = end

    def value(self, me):
        """Heuristic terminal value: army plus production, relative to the strongest rival."""
        strength = {p: 0.0 for p in self.players}
        alive = set()
        for i, b in enumerate(self.state.bases):
            p = self.owners[i]
            if p:
                strength[p] += self.units[i] + 28 * b.production_rate + 3
                alive.add(p)
        for _, f in self.fleets:
            strength[f.owner] += f.units
            alive.add(f.owner)
        if me not in alive:
            return -1000000.0
        if alive == {me}:
            return 1000000.0
        enemy = [v for p, v in strength.items() if p != me]
        return max(-99999.0, min(99999.0, strength[me] - max(enemy, default=0) - 0.2 * sum(enemy)))
