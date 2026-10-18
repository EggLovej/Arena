"""Fixed validation and untouched generalization suites for trained checkpoints."""

from __future__ import annotations

import itertools
import time

import numpy as np

from ricochet.game.config import RulesConfig
from ricochet.game.engine import GameEngine
from ricochet.game.maps import jitter_spawns, load_map
from ricochet.game.types import EventKind
from ricochet.runtime.match import MatchSession, safe_load_controller

# Training and validation share these maps; the test suite keeps its maps and pursuit bot
# unseen during training.
TRAINING_MAPS = ("classic", "crossfire")
TEST_MAPS = ("channels",)
SCRIPTED = ("random_ai", "template_ai", "direct_ai", "bank_shot_ai", "reactive_defender_ai")
TEST_OPPONENTS = ("close_range_ai", "bank_shot_ai", "reactive_defender_ai")


def measured_match(first, second, map_id, seed=0, jitter=False):
    rules = RulesConfig()
    definition = load_map(map_id, rules=rules)
    if jitter:
        definition = jitter_spawns(definition, rules, np.random.default_rng(seed))
    session = MatchSession(
        GameEngine(definition, rules, seed, False),
        (
            safe_load_controller(first) if isinstance(first, str) else first,
            safe_load_controller(second) if isinstance(second, str) else second,
        ),
        (
            first if isinstance(first, str) else "checkpoint",
            second if isinstance(second, str) else "checkpoint",
        ),
    )
    metrics = {
        key: [0, 0]
        for key in ("shots", "blocks", "shield_activations", "damage_taken", "energy_spent")
    }
    started = time.perf_counter()
    while not session.finished:
        before = [session.engine.tanks[i].energy for i in (0, 1)]
        before_health = [session.engine.tanks[i].health for i in (0, 1)]
        result = session.step()
        for i in (0, 1):
            available = min(rules.max_energy, before[i] + rules.energy_regen_per_second * rules.dt)
            metrics["energy_spent"][i] += max(0, available - session.engine.tanks[i].energy)
            metrics["damage_taken"][i] += before_health[i] - session.engine.tanks[i].health
        for event in result.events:
            key = {
                EventKind.FIRED: "shots",
                EventKind.BLOCKED: "blocks",
                EventKind.SHIELD_ACTIVATED: "shield_activations",
            }.get(event.kind)
            if key is not None:
                metrics[key][event.player_id] += 1
    record = session.summary().to_dict()
    record.update(metrics)
    record.update(jitter=jitter, wall_seconds=time.perf_counter() - started)
    return record


def evaluate(controller, suite="validation", quick=False):
    maps = TEST_MAPS if suite == "test" else TRAINING_MAPS
    opponents = TEST_OPPONENTS if suite == "test" else SCRIPTED
    scenarios = [(0, False)] + [(s, True) for s in range(10000, 10010)]
    if quick:
        opponents = ("direct_ai",)
        scenarios = [(0, False)]
    rows = []
    for map_id, opponent, (seed, jitter), seat in itertools.product(
        maps, opponents, scenarios, (0, 1)
    ):
        players = [opponent, opponent]
        players[seat] = controller
        row = measured_match(*players, map_id, seed, jitter=jitter)
        row.update(learner_seat=seat, opponent=opponent)
        rows.append(row)
    valid = [r for r in rows if not r["invalid"]]
    wins = sum(r["winner"] == r["learner_seat"] for r in valid)
    draws = sum(r["winner"] is None for r in valid)
    return {
        "suite": suite,
        "quick": quick,
        "games": len(rows),
        "valid_games": len(valid),
        "score": (wins + 0.5 * draws) / len(valid) if valid else None,
        "wins": wins,
        "draws": draws,
        "losses": len(valid) - wins - draws,
        "matches": rows,
    }
