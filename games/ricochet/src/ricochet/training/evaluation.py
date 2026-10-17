"""Fixed validation and untouched generalization suites, plus shield comparisons."""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from ricochet.game.config import RulesConfig
from ricochet.game.engine import GameEngine
from ricochet.game.maps import available_maps, jitter_spawns, load_map
from ricochet.game.types import EventKind
from ricochet.runtime.match import MatchSession, safe_load_controller

# Training and validation share these maps; the test suite keeps its maps and pursuit bot
# unseen during training.
TRAINING_MAPS = ("classic", "crossfire")
TEST_MAPS = ("channels",)
SCRIPTED = ("random_ai", "template_ai", "direct_ai", "bank_shot_ai", "reactive_defender_ai")
TEST_OPPONENTS = ("close_range_ai", "bank_shot_ai", "reactive_defender_ai")
BALANCE_BOTS = (
    "template_ai",
    "direct_ai",
    "bank_shot_ai",
    "reactive_defender_ai",
    "close_range_ai",
)


def measured_match(first, second, map_id, seed=0, rules=None, jitter=False):
    rules = rules or RulesConfig()
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


def compare_shields(output: Path):
    rows = []
    for preset, map_id, pair, swap in itertools.product(
        ("legacy_shield", "default"),
        available_maps(),
        itertools.combinations(BALANCE_BOTS, 2),
        (False, True),
    ):
        names = pair[::-1] if swap else pair
        record = measured_match(*names, map_id, rules=RulesConfig.preset(preset))
        record["preset"] = preset
        rows.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def benchmark(steps=10000):
    from ricochet.training.gym_env import RicochetEnv

    results = {}
    for verification in (True, False):
        engine = GameEngine(load_map("classic"), verification=verification)
        started = time.perf_counter()
        for _ in range(steps):
            if engine.terminated:
                engine = GameEngine(load_map("classic"), verification=verification)
            engine.step({})
        elapsed = time.perf_counter() - started
        results[f"engine_verification_{verification}"] = {
            "steps": steps,
            "seconds": elapsed,
            "steps_per_second": steps / elapsed,
        }
    for label, opponent in [("gym", "bank_shot_ai")]:
        env = RicochetEnv(opponent=opponent, seat_mode=0)
        env.reset(seed=0)
        action = np.zeros(6, dtype=np.float32)
        started = time.perf_counter()
        for _ in range(steps):
            _, _, done, truncated, _ = env.step(action)
            if done or truncated:
                env.reset()
        elapsed = time.perf_counter() - started
        results[label] = {"steps": steps, "seconds": elapsed, "steps_per_second": steps / elapsed}
    match = measured_match("direct_ai", "bank_shot_ai", "classic")
    results["scripted_match"] = {
        "ticks": match["ticks"],
        "seconds": match["wall_seconds"],
        "steps_per_second": match["ticks"] / match["wall_seconds"],
    }
    return results
