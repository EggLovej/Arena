"""Atlas tactical policies and exact committed-arrival forecasts.

Only the public GameView is used. No map names or coordinates affect planning.
Speculative reserves and investment scores are heuristics; event combat follows
the game rules. See AI_RESEARCH_NOTES.md for experiments and limitations."""
from collections import defaultdict
from heapq import heappop, heappush
from math import ceil, floor
from functools import lru_cache
from game_types import SendAction
DEFAULTS = dict(campaign_funding=False, horizon=24.0, reserve=1.0, buffer=0.0, logistics=1.0, enemy=1.8, expansion=1.0, coordination=True, saving=False, costscale=5.0, holding=0.65, evacuation=True, ffa=True, surge=0.0, rally=0.0, overwhelm=1.0)

def arrival_ticks(seconds, dt):
    """First engine tick at which a fleet reaches its destination (including tolerance)."""
    return max(1, ceil((seconds - 1e-09) / dt))

def resolve(owner, units, arrivals):
    """Pool each owner separately; keep the largest force minus the runner-up."""
    forces = dict(arrivals)
    if units > 1e-09:
        forces[owner] = forces.get(owner, 0.0) + units
    ranked = sorted(forces.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return (0, 0.0)
    other = ranked[1][1] if len(ranked) > 1 else 0.0
    delta = ranked[0][1] - other
    return (0, 0.0) if abs(delta) <= 1e-09 else (ranked[0][0], delta)

class Position:
    """Indexed immutable observation with chronological, per-owner arrival events."""

    def __init__(self, state):
        self.state = state
        self.me = state.player_id
        self.dt = state.seconds_per_tick
        self.bases = state.bases
        self.index = {b.id: i for i, b in enumerate(self.bases)}
        self.n = len(self.bases)
        self.graph = [[] for _ in self.bases]
        for e in state.edges:
            a, b = (self.index[e.a], self.index[e.b])
            t = arrival_ticks(e.travel_time, self.dt)
            self.graph[a].append((b, t))
            self.graph[b].append((a, t))
        events = [defaultdict(lambda: defaultdict(float)) for _ in self.bases]
        for f in state.fleets:
            events[self.index[f.target_id]][arrival_ticks(f.remaining_time, self.dt)][f.owner] += f.units
        self.events = [sorted(e.items()) for e in events]
        self.mine = [i for i, b in enumerate(self.bases) if b.owner == self.me]
        self.cache = {}

    def project(self, i, ticks, removed=0, extra=None):
        """Forecast this base with no future departures, optionally adding hypothetical arrivals."""
        key = (i, ticks)
        if not removed and extra is None and (key in self.cache):
            return self.cache[key]
        b = self.bases[i]
        owner, units, last = (b.owner, b.units - removed, 0)
        events = self.events[i]
        if extra:
            merged = {t: dict(a) for t, a in events if t <= ticks}
            for t, p, u in extra:
                if t <= ticks:
                    a = merged.setdefault(t, {})
                    a[p] = a.get(p, 0) + u
            events = sorted(merged.items())
        for t, arrivals in events:
            if t > ticks:
                break
            if owner:
                units += (t - last) * self.dt * b.production_rate
            owner, units = resolve(owner, units, arrivals)
            last = t
        if owner:
            units += (ticks - last) * self.dt * b.production_rate
        if not removed and extra is None:
            self.cache[key] = (owner, units)
        return (owner, units)

    def available(self, i):
        """Maximum send preserving ownership through every committed arrival."""
        b = self.bases[i]
        balance, reserve, last = (0.0, 0.0, 0)
        for t, forces in self.events[i]:
            balance += (t - last) * self.dt * b.production_rate + forces.get(self.me, 0)
            balance -= max((u for p, u in forces.items() if p != self.me), default=0)
            reserve = max(reserve, -balance + 1e-07)
            last = t
        return max(0, min(int(b.units), floor(b.units - reserve)))

    def required(self, i, t):
        """Smallest integer send that captures at this tick, pooling simultaneous arrivals."""
        owner, units = self.project(i, t)
        if owner == self.me:
            return 0
        b = self.bases[i]
        before_owner, before_units = self.project(i, t - 1)
        if before_owner:
            before_units += b.production_rate * self.dt
        forces = dict(next((a for at, a in self.events[i] if at == t), {}))
        if before_units > 1e-09:
            forces[before_owner] = forces.get(before_owner, 0) + before_units
        opposition = max((u for p, u in forces.items() if p != self.me), default=0)
        return max(0, floor(opposition - forces.get(self.me, 0) + 1e-09) + 1)

@lru_cache(maxsize=16)
def _position(state):
    return Position(state)

def policy(state, **overrides):
    """Rank rescue, evacuation, capture, coordinated attack, saving, and logistics plans.

Defaults form a fast policy; the controller searches a portfolio of overrides.
Saving suppresses only moves from the source saving for that investment.
Logistics uses shortest friendly routes and avoids already-funded campaigns."""
    cfg = DEFAULTS | overrides
    p = _position(state)
    if not p.mine:
        return None
    me, bs, graph = (p.me, p.bases, p.graph)
    budget = {i: p.available(i) for i in p.mine}
    plans = []
    savings = {}
    horizon = cfg['horizon']
    strength = defaultdict(float)
    for b in bs:
        if b.owner and b.owner != me:
            strength[b.owner] += b.units + horizon * b.production_rate
    for f in state.fleets:
        if f.owner != me:
            strength[f.owner] += f.units
    largest = max(strength.values(), default=1.0)
    weights = {q: (u / max(1.0, largest)) ** 2 for q, u in strength.items()}

    def swing(b):
        if not b.owner:
            return cfg['expansion']
        return cfg['enemy'] * (1 + weights.get(b.owner, 1.0)) / 2 if cfg['ffa'] else cfg['enemy']

    def cost(b):
        if not b.owner:
            return b.units * 0.65
        return b.units * (1 - weights.get(b.owner, 1.0)) if cfg['ffa'] else 0.0

    def risk(i, amount):
        b = bs[i]
        left = b.units - amount
        danger = max((max(0.0, bs[j].units - left - b.production_rate * t * p.dt) for j, t in graph[i] if bs[j].owner not in (0, me)), default=0)
        exposure = max((max(0.0, bs[j].units - left - b.production_rate * t * p.dt) / max(1.0, bs[j].units) for j, t in graph[i] if bs[j].owner not in (0, me)), default=0)
        return cfg['reserve'] * (danger + b.production_rate * horizon * exposure)

    def add(score, i, j, amount):
        amount = min(int(bs[i].units), int(amount))
        if amount > 0:
            plans.append((score, i, j, amount))
    for j in p.mine:
        hostile = [t for t, a in p.events[j] if any((q != me for q in a))]
        if not hostile:
            continue
        end = max(hostile)
        if all((p.project(j, t)[0] == me for t in hostile)):
            continue
        for i, t in graph[j]:
            if i not in budget or budget[i] <= 0:
                continue
            h = max(t, end)
            lo, hi = (1, budget[i])
            if p.project(j, h, extra=[(t, me, hi)])[0] != me:
                continue
            while lo < hi:
                mid = (lo + hi) // 2
                if p.project(j, h, extra=[(t, me, mid)])[0] == me:
                    hi = mid
                else:
                    lo = mid + 1
            amount = min(budget[i], lo + 1)
            score = (bs[j].production_rate * horizon * 2 + min(bs[j].units, 20)) / (1 + t * p.dt / 6) - risk(i, amount)
            add(score, i, j, amount)
    if cfg['evacuation']:
        for i in p.mine:
            if budget[i] or bs[i].units < 1:
                continue
            losses = [t for t, a in p.events[i] if p.project(i, t)[0] != me]
            if not losses:
                continue
            for j, t in graph[i]:
                if bs[j].owner == me:
                    add(10 + bs[i].units * 0.2 - t * p.dt, i, j, int(bs[i].units))
                elif p.required(j, t) <= int(bs[i].units):
                    add(12 + bs[j].production_rate * 5, i, j, int(bs[i].units))
    for j, b in enumerate(bs):
        if b.owner == me:
            continue
        sources = [(i, t, budget[i]) for i, t in graph[j] if i in budget and budget[i] > 0]
        for i, t, available in sources:
            need = p.required(j, t)
            if not need:
                continue
            if need > available:
                if cfg['saving'] and bs[i].production_rate > 0:
                    wait = (need - available) / max(0.01, bs[i].production_rate - (b.production_rate if b.owner else 0))
                    if wait < horizon and (not any((q not in (0, me) for _, a in p.events[i] for q in a))):
                        gain = b.production_rate * horizon * swing(b) + 2
                        access = sum((bs[k].production_rate / (1 + bs[k].units / 12) for k, _ in graph[j] if bs[k].owner != me))
                        score = (gain + 2 * access - cost(b)) / (1 + (t * p.dt + wait) / 5 + need / cfg['costscale'])
                        savings[i] = max(savings.get(i, 0), score)
                continue
            dt = t * p.dt
            threat = max((bs[k].units * cfg['holding'] - b.production_rate * at * p.dt for k, at in graph[j] if k != i and bs[k].owner not in (0, me)), default=0)
            extra = max(ceil(cfg['buffer'] * b.production_rate), ceil(threat))
            amount = max(ceil(need * (cfg['overwhelm'] if b.owner else 1.0)), need + max(0, extra), ceil(available * cfg['surge']))
            if amount > available:
                continue
            later = [(at, a) for at, a in p.events[j] if at > t and any((q != me for q in a))]
            if later and p.project(j, later[-1][0], extra=[(t, me, amount)])[0] != me:
                lo, hi = (amount, available)
                if p.project(j, later[-1][0], extra=[(t, me, hi)])[0] != me:
                    continue
                while lo < hi:
                    mid = (lo + hi) // 2
                    if p.project(j, later[-1][0], extra=[(t, me, mid)])[0] == me:
                        hi = mid
                    else:
                        lo = mid + 1
                amount = lo
            gain = b.production_rate * horizon * swing(b)
            access = sum((bs[k].production_rate / (1 + bs[k].units / 12) for k, _ in graph[j] if bs[k].owner != me))
            gain += 2 * access + 2
            loss = cost(b) / 0.65
            score = (gain - loss * 0.65) / (1 + dt / 5 + need / cfg['costscale']) - risk(i, amount)
            add(score, i, j, amount)
        if cfg['coordination'] and len(sources) > 1:
            end = max((t for _, t, _ in sources)) + len(sources)
            extras = [(t + k, me, u) for k, (i, t, u) in enumerate(sorted(sources, key=lambda x: -x[1]))]
            if p.project(j, end, extra=extras)[0] == me and p.project(j, end)[0] != me:
                i, t, amount = max(sources, key=lambda x: (x[1], x[2]))
                total = sum((u for _, _, u in sources))
                gain = b.production_rate * horizon * swing(b) + 2
                score = (gain - cost(b)) / (1 + end * p.dt / 5 + total / cfg['costscale']) - risk(i, amount)
                add(score, i, j, amount)
    for j, b in enumerate(bs):
        if b.owner == me:
            continue
        dist = {}
        heap = []
        for i, t in graph[j]:
            if i in budget:
                dist[i] = (t, j)
                heappush(heap, (t, i, j))
        while heap:
            d, i, hop = heappop(heap)
            if dist[i] != (d, hop):
                continue
            for k, t in graph[i]:
                if k not in budget:
                    continue
                nd = d + t
                if k not in dist or nd < dist[k][0]:
                    dist[k] = (nd, i)
                    heappush(heap, (nd, k, i))
        for i, (d, hop) in dist.items():
            amount = budget[i]
            if hop == j or amount < 2:
                continue
            need = p.required(j, d)
            if not need:
                continue
            if cfg['campaign_funding']:
                threat = max((bs[k].units * cfg['holding'] - b.production_rate * at * p.dt
                              for k, at in graph[j] if bs[k].owner not in (0, me)), default=0.)
                need = max(ceil(need * (cfg['overwhelm'] if b.owner else 1.)),
                           need + max(0, ceil(cfg['buffer'] * b.production_rate), ceil(threat)))
            local = sum((budget[k] for k, _ in graph[j] if k in budget))
            incoming = sum((u.get(me, 0) for k, _ in graph[j] if k in budget for _, u in p.events[k]))
            if local + incoming >= need + 4:
                continue
            amount = min(amount, max(2, need + 3 - local - int(incoming), ceil(amount * cfg['rally'])))
            gain = b.production_rate * horizon * swing(b) + 2
            score = cfg['logistics'] * (gain - cost(b)) / (1 + d * p.dt / 5 + need / cfg['costscale'])
            score *= min(1.0, amount / max(3.0, need - local - incoming))
            score -= risk(i, amount)
            add(score, i, hop, amount)
    plans = [x for x in plans if x[0] >= savings.get(x[1], 0)]
    if not plans:
        return None
    score, i, j, amount = max(plans, key=lambda x: x[0])
    if score <= 0 or amount == 0:
        return None
    return SendAction(bs[i].id, bs[j].id, amount)
