"""Atlas: public-observation opponent modelling and policy search.

Known scripted opponents use chronological forecasts and conditional winning
continuations. A plausibly matched learned opponent uses independent sampled
60-second futures, nine candidate policies and three samples per policy.
Three-player openings spend one tick observing before committing an army.

The public neural checkpoint is pinned at match start; no running opponent
object, private random generator, map name, or screen coordinates are read.
Contexts are bounded. Use separate processes for independent interleaved
matches, since GameView has no match identifier and the model is process-local.
See AI_HANDOFF.md for measurements, dependencies and remaining limitations.
"""
from dataclasses import replace
from functools import partial
from collections import OrderedDict
from game_types import SendAction
from ais._atlas_sim import Simulation
from ais import aggressive_ai, careful_ai, strategic_ai
from ais.atlas_clean_ai import CAUTIOUS, STYLE_POLICIES, STYLES, action_distance, policy

MODELS = [strategic_ai.choose_action, careful_ai.choose_action, aggressive_ai.choose_action, *STYLE_POLICIES]
_CONTEXTS = OrderedDict()
FINE_HORIZON = 12.0
SEARCH_HORIZON = 30.0
REPLAN_SECONDS = 1.0

def choose_action(state):
    """Choose one move using only the public observation and prior public observations."""
    key = (state.player_id, state.seconds_per_tick, tuple(((b.id, b.production_rate) for b in state.bases)), state.edges)
    ctx = _CONTEXTS.get(key)
    if ctx and ctx['previous'] == state:
        return ctx['action']
    if ctx is None or state.tick != ctx['previous'].tick + 1:
        ctx = dict(previous=state, errors={}, config=STYLES[0], next_plan=state.tick, action=None, observed={}, neural_possible={}, neural_fit={}, certified=None, certify_after=0)
        from ais._atlas_model_planner import configure_checkpoint
        configure_checkpoint()
        _CONTEXTS[key] = ctx
        while len(_CONTEXTS) > 12:
            _CONTEXTS.popitem(last=False)
        if state.tick == 0 and len({b.owner for b in state.bases} - {0}) > 2:
            # Spend one tick observing before committing an opening army.
            return None
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
                if min(errors) > .01 and ctx['neural_possible'].get(enemy, True):
                    from ais._atlas_model_planner import evidence
                    fit=evidence(view,actual,estimates)
                    if fit is None:
                        ctx['neural_possible'][enemy]=False
                        ctx['next_plan']=state.tick
                    else:
                        ctx['neural_fit'][enemy]=.97*ctx['neural_fit'].get(enemy,0.)+fit
    if ctx['certified'] is None and state.tick >= ctx['next_plan']:
        players = sorted(set((b.owner for b in state.bases)) - {0} | {f.owner for f in state.fleets})
        models = {p: MODELS[min(range(len(MODELS)), key=lambda i: ctx['errors'].get(p, [0.0] * len(MODELS))[i])] for p in players if p != state.player_id}
        unfamiliar = any((min(range(len(MODELS)), key=lambda i: ctx['errors'].get(p, [0.0] * len(MODELS))[i]) >= 3 or min(ctx['errors'].get(p, [0.0])) > 0.01 for p in models))
        neural_players={p for p in models if min(ctx['errors'].get(p,[0.])) > .01
                        and ctx['neural_possible'].get(p,True)
                        and ctx['neural_fit'].get(p,0.) > -10.}
        if unfamiliar and neural_players:
            from ais._atlas_model_planner import select
            opponents={p:None if p in neural_players else model for p,model in models.items()}
            ctx['config'] = select(state,opponents)
            ctx['next_plan'] = state.tick + max(1, round(3. / state.seconds_per_tick))
            config = dict(ctx['config'])
            campaign = config.pop('campaign', False)
            action = policy(state, campaign_funding=campaign, **config)
            ctx['previous'] = state
            ctx['action'] = action
            return action
        scenarios = [models]
        uncertain = [p for p in models if min(ctx['errors'].get(p, [0.0])) > 6.0]
        if uncertain:
            alternative = dict(models)
            for p in uncertain:
                alternative[p] = partial(policy, **CAUTIOUS)
            scenarios.append(alternative)
        step = max(1, round(0.5 / state.seconds_per_tick))
        durations = [1] * max(1, round(FINE_HORIZON / state.seconds_per_tick)) + [step] * max(1, round((SEARCH_HORIZON - FINE_HORIZON) / (step * state.seconds_per_tick)))
        nsteps = len(durations)
        best = (-float('inf'), ctx['config'])
        rankings = []
        for cfg in STYLES:
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
    config = dict(ctx['config'])
    campaign = config.pop('campaign', False)
    action = policy(state, campaign_funding=campaign, **config)
    ctx['previous'] = state
    ctx['action'] = action
    return action
