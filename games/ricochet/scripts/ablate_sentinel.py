"""Controlled no-bank/no-dodge comparisons, using the normal match session."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from ricochet.game.engine import GameEngine
from ricochet.game.maps import load_map
from ricochet.runtime.loading import load_controller
from ricochet.runtime.match import MatchSession
from ricochet.runtime.tournament import head_to_head, write_results


class Variant:
    def __init__(self, variant):
        self.variant = variant
        self.bot = load_controller("sentinel_ai")

    def reset(self, context):
        self.bot.reset(context)
        self.bot.use_banks = self.variant != "no_banks"
        if self.variant == "no_dodge":
            choose = self.bot.choose_move
            self.bot.choose_move = lambda obs: choose(replace(obs, projectiles=()))

    def act(self, obs):
        return self.bot.act(obs)


def play(fixture):
    variant, opponent, map_id, seat = fixture
    bots = [Variant(variant), load_controller(opponent)]
    names = ["sentinel_ai", opponent]
    if seat:
        bots.reverse()
        names.reverse()
    session = MatchSession(
        GameEngine(load_map(map_id), verification=False), tuple(bots), tuple(names)
    )
    while not session.finished:
        session.step()
    return session.summary()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    output = Path("results/sentinel_ablation")
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for variant in ("full", "no_banks", "no_dodge"):
            fixtures = [
                (variant, opponent, map_id, seat)
                for opponent in ("direct_ai", "bank_shot_ai", "close_range_ai")
                for map_id in ("classic", "generated:20", "generated:21")
                for seat in (0, 1)
            ]
            results = list(pool.map(play, fixtures))
            write_results(results, output / f"{variant}.json")
            table = {b: r for (a, b), r in head_to_head(results).items() if a == "sentinel_ai"}
            report[variant] = table
            print(variant, table, flush=True)
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
