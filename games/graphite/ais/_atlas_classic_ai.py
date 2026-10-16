"""Atlas: opponent-adaptive receding-horizon policy search.

Replans a 30-second future once per simulated second, with full tick decisions
for the first 12 seconds. Each real tick still receives its own legal action.
Fits thirteen opponent hypotheses using visible fleet departures; unfamiliar
behavior adds a second scenario. The bundled AIs are explicit hypotheses,
not privileged access to the controllers running in the match.
Keeps a fully simulated winning continuation while observed moves match its
models; unfamiliar behavior receives additional full-frequency refinement.

Context is bounded and separated by player, topology, and tick duration. A
new match, discontinuous tick, or duplicate observation is handled explicitly.
For interleaved independent matches sharing identical maps and seats, use a
separate module/process per match; GameView has no match identifier."""
from dataclasses import replace
from functools import partial
from collections import OrderedDict
from ais._atlas_policy import policy as base_policy, _position
from math import ceil
from game_types import SendAction
from ais._atlas_sim import Simulation
from ais import aggressive_ai, careful_ai, strategic_ai

def policy(state, **cfg):
    action = base_policy(state, **cfg)
    if action is None:
        return None
    p = _position(state)
    i, j = (p.index[action.source_id], p.index[action.target_id])
    source, target = (p.bases[i], p.bases[j])
    if target.owner == 0 and target.production_rate > source.production_rate and any((p.bases[k].owner not in (0, state.player_id) for k, _ in p.graph[j])):
        t = next((t for k, t in p.graph[i] if k == j))
        need = p.required(j, t)
        available = p.available(i)
        if need and available >= action.units:
            amount = min(available, max(action.units, need + ceil(target.production_rate)))
            return SendAction(action.source_id, action.target_id, amount)
    return action
CONFIGS = [dict(reserve=0.1, buffer=0.0, saving=False, holding=0.0, costscale=20.0, surge=1.0, rally=1.0, overwhelm=2.0, enemy=3.0), dict(reserve=0.0, buffer=0.0, saving=False, holding=0.0, costscale=20.0, surge=1.0, rally=1.0, enemy=3.0), dict(reserve=0.0, buffer=0.0, saving=False, holding=0.0, costscale=20.0, surge=1.0, enemy=3.0), dict(reserve=0.5, buffer=0.0, saving=False, holding=0.0, costscale=10.0, surge=0.9), dict(reserve=1.0, buffer=0.0, saving=False, holding=0.0, costscale=10.0, surge=0.6), dict(reserve=1.0, buffer=0.0, saving=False, holding=0.0, costscale=5.0), dict(reserve=1.0, buffer=0.0, saving=False, holding=0.65, costscale=5.0), dict(reserve=0.5, buffer=0.0, saving=False, holding=0.25, costscale=20.0), dict(reserve=1.0, buffer=0.0, saving=True, holding=0.25, costscale=10.0, horizon=40.0), dict(reserve=2.0, buffer=2.0, saving=True, holding=0.5, costscale=20.0)]
MODELS = [strategic_ai.choose_action, careful_ai.choose_action, aggressive_ai.choose_action, *[partial(policy, **cfg) for cfg in CONFIGS]]
_CONTEXTS = OrderedDict()
FINE_HORIZON = 12.0
SEARCH_HORIZON = 30.0
REPLAN_SECONDS = 1.0

def action_distance(a, b):
    """Prediction error for a visible departed fleet, including target and army size."""
    if a == b:
        return 0.0
    if a is None or b is None:
        return 1.0
    if (a.source_id, a.target_id) != (b.source_id, b.target_id):
        return 1.0
    return min(1.0, abs(a.units - b.units) / max(1, a.units, b.units))

def choose_action(state):
    """Choose one move using only the public observation and prior public observations."""
    from game_types import SendAction
    key = (state.player_id, state.seconds_per_tick, tuple(((b.id, b.production_rate) for b in state.bases)), state.edges)
    ctx = _CONTEXTS.get(key)
    if ctx and ctx['previous'] == state:
        return ctx['action']
    if ctx is None or state.tick != ctx['previous'].tick + 1:
        ctx = dict(previous=state, errors={}, config=CONFIGS[0], next_plan=state.tick, action=None, observed={}, certified=None, certify_after=0)
        _CONTEXTS[key] = ctx
        while len(_CONTEXTS) > 12:
            _CONTEXTS.popitem(last=False)
    else:
        prev = ctx['previous']
        for enemy in set((b.owner for b in prev.bases)) - {0, state.player_id}:
            if min((e.travel_time for e in state.edges), default=0) <= state.seconds_per_tick:
                continue
            recent = [f for f in state.fleets if f.owner == enemy and abs(f.total_time - f.remaining_time - state.seconds_per_tick) < 1e-07]
            actual = SendAction(recent[0].source_id, recent[0].target_id, recent[0].units) if recent else None
            view = replace(prev, player_id=enemy)
            if actual is not None:
                ctx['observed'][enemy] = ctx['observed'].get(enemy, 0) + 1
            if ctx['certified'] and enemy in ctx['certified'] and (ctx['certified'][enemy](view) != actual):
                ctx['certified'] = None
                ctx['next_plan'] = state.tick
            estimates = [model(view) for model in MODELS]
            if actual is not None or any((a is not None for a in estimates)):
                errors = ctx['errors'].setdefault(enemy, [0.0] * len(MODELS))
                for i, a in enumerate(estimates):
                    errors[i] = errors[i] * 0.97 + action_distance(a, actual)
    if ctx['certified'] is None and state.tick >= ctx['next_plan']:
        players = sorted(set((b.owner for b in state.bases)) - {0} | {f.owner for f in state.fleets})
        models = {p: MODELS[min(range(len(MODELS)), key=lambda i: ctx['errors'].get(p, [0.0] * len(MODELS))[i])] for p in players if p != state.player_id}
        unfamiliar = any((min(range(len(MODELS)), key=lambda i: ctx['errors'].get(p, [0.0] * len(MODELS))[i]) >= 3 or min(ctx['errors'].get(p, [0.0])) > 0.01 for p in models))
        scenarios = [models]
        uncertain = [p for p in models if min(ctx['errors'].get(p, [0.0])) > 6.0]
        if uncertain:
            alternative = dict(models)
            for p in uncertain:
                alternative[p] = partial(policy, reserve=1.0, holding=0.3, buffer=0.0, saving=False, surge=0.6)
            scenarios.append(alternative)
        step = max(1, round(0.5 / state.seconds_per_tick))
        durations = [1] * max(1, round(FINE_HORIZON / state.seconds_per_tick)) + [step] * max(1, round((SEARCH_HORIZON - FINE_HORIZON) / (step * state.seconds_per_tick)))
        nsteps = len(durations)
        best = (-float('inf'), ctx['config'])
        rankings = []
        for cfg in CONFIGS:
            outcomes = []
            for opponent_models in scenarios:
                sim = Simulation(state)
                integral = 0
                for k, duration in enumerate(durations):
                    actions = {p: policy(sim.view(p), **cfg) if p == state.player_id else opponent_models[p](sim.view(p)) for p in players}
                    sim.step(actions, duration)
                    value = sim.value(state.player_id)
                    if abs(value) > 100000.0:
                        value += (1 if value > 0 else -1) * (SEARCH_HORIZON - sim.tick * state.seconds_per_tick) * 100
                        break
                    integral += value / nsteps
                outcomes.append(value + 0.1 * integral)
            score = 0.65 * sum(outcomes) / len(outcomes) + 0.35 * min(outcomes)
            rankings.append((score, cfg))
            if score > best[0]:
                best = (score, cfg)
        if unfamiliar:
            best = (-float('inf'), ctx['config'])
            for _, cfg in sorted(rankings, key=lambda x: x[0], reverse=True)[:3]:
                outcomes = []
                nsteps = max(1, round(SEARCH_HORIZON / state.seconds_per_tick))
                for opponent_models in scenarios:
                    sim = Simulation(state)
                    integral = 0
                    for _ in range(nsteps):
                        actions = {p: policy(sim.view(p), **cfg) if p == state.player_id else opponent_models[p](sim.view(p)) for p in players}
                        sim.step(actions, 1)
                        value = sim.value(state.player_id)
                        if abs(value) > 100000.0:
                            value += (1 if value > 0 else -1) * (SEARCH_HORIZON - sim.tick * state.seconds_per_tick) * 100
                            break
                        integral += value / nsteps
                    outcomes.append(value + 0.1 * integral)
                score = 0.65 * sum(outcomes) / len(outcomes) + 0.35 * min(outcomes)
                if score > best[0]:
                    best = (score, cfg)
        if not unfamiliar and state.tick >= ctx['certify_after'] and all((ctx['observed'].get(p, 0) >= 1 and min(ctx['errors'].get(p, [1.0])) < 1e-12 and (ctx['observed'].get(p, 0) >= 2 or sum((e < 1e-12 for e in ctx['errors'].get(p, []))) == 1) for p in models)):
            best_win_steps = max(1, round(max(60.0, 180.0 - state.time_seconds) / state.seconds_per_tick))
            for _, cfg in sorted(rankings, key=lambda x: x[0], reverse=True):
                sim = Simulation(state)
                certified = False
                for _ in range(best_win_steps):
                    actions = {p: policy(sim.view(p), **cfg) if p == state.player_id else models[p](sim.view(p)) for p in players}
                    sim.step(actions, 1)
                    alive = set(sim.owners) - {0} | {f.owner for _, f in sim.fleets}
                    if alive == {state.player_id}:
                        certified = True
                        break
                    if state.player_id not in alive:
                        break
                if certified:
                    best = (float('inf'), cfg)
                    ctx['certified'] = dict(models)
                    best_win_steps = sim.tick - 1
            ctx['certify_after'] = state.tick + max(1, round(5.0 / state.seconds_per_tick))
        ctx['config'] = best[1]
        ctx['next_plan'] = state.tick + max(1, round(REPLAN_SECONDS / state.seconds_per_tick))
    action = policy(state, **ctx['config'])
    ctx['previous'] = state
    ctx['action'] = action
    return action
