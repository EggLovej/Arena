"""Seat-swapped scripted-bot evaluation using the repository fixture runner.

Example: PYTHONPATH=src python scripts/evaluate_sentinel.py --suite heldout --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from ricochet.game.config import RulesConfig
from ricochet.runtime.loading import available_ais, default_ais_directory
from ricochet.runtime.tournament import head_to_head, play_fixtures, write_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("authored", "heldout"), default="authored")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/sentinel"))
    args = parser.parse_args()
    if args.rounds < 1 or args.workers < 1:
        parser.error("rounds and workers must be positive")
    rules = RulesConfig()
    presets = {"default": rules}
    if args.suite == "heldout":
        presets.update(
            nerfed_shield=replace(rules, shield_activation_cost=60, shield_energy_per_second=90),
            disabled_shield=replace(rules, shield_activation_cost=rules.max_energy + 1),
        )
    maps = (
        ["classic", "crossfire", "channels"]
        if args.suite == "authored"
        else [f"generated:{s}" for s in range(args.seed, args.seed + args.rounds)]
    )
    bots = [b for b in available_ais() if b != "sentinel_ai"]
    report = {"suite": args.suite, "maps": maps, "rules": {}, "results": {}, "source_sha256": {}}
    for bot in ["sentinel_ai"] + bots:
        report["source_sha256"][bot] = hashlib.sha256(
            (default_ais_directory() / f"{bot}.py").read_bytes()
        ).hexdigest()
    for preset, config in presets.items():
        fixtures = [
            (*pair, m, args.seed, config)
            for m in maps
            for bot in bots
            for pair in [("sentinel_ai", bot), (bot, "sentinel_ai")]
        ]
        results = play_fixtures(
            fixtures,
            args.workers,
            lambda n, total, preset=preset: print(f"{preset}: {n}/{total}", flush=True),
        )
        write_results(results, args.output / f"{args.suite}_{preset}.json")
        records = {
            opponent: list(record)
            for (bot, opponent), record in head_to_head(results).items()
            if bot == "sentinel_ai"
        }
        report["results"][preset] = {
            "opponents_wdl": records,
            "total_wdl": [sum(r[i] for r in records.values()) for i in range(3)],
            "faults": sum(sum(r.faults) for r in results),
            "mean_match_action_ms": sum(
                r.mean_action_ms[r.controllers.index("sentinel_ai")] for r in results
            )
            / len(results),
            "max_action_ms": max(
                r.max_action_ms[r.controllers.index("sentinel_ai")] for r in results
            ),
        }
        report["rules"][preset] = config.to_dict()
        print(json.dumps(report["results"][preset], indent=2), flush=True)
        (args.output / f"{args.suite}_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
