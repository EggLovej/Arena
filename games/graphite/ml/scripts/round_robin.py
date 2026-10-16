"""Networks against each other and against scripted AIs, every pairing, with the scores updated live.

Every network draws its moves at one temperature (``--temperature``; Beast and Neural AI play at
0.2) and has no send limit, whatever it trained under. Each pairing plays ``--games`` games on the
comparisons' fresh maps (`ml compare`), both seats each. At the end: each pairing's score with its
standard error, ratings fitted to all of it (strategic_ai = 1000 when it plays), and everything in
ml/runs/logs/round_robin_<name>.json.

    .venv/bin/python ml/scripts/round_robin.py --name free2 \\
        --network "free_gap0 r50=ml/runs/free_gap0/round_0050.pt" --network "new=ml/runs/x/round_0050.pt" \\
        --opponents strategic_ai --temperature 0.2 --games 200
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIRST_COMPARISON_MAP = 300  # as in ml/evaluation/compare.py
REFERENCE = "strategic_ai"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="Names the output file")
    parser.add_argument("--network", action="append", required=True, help="LABEL=PATH of a checkpoint; repeat")
    parser.add_argument("--opponents", nargs="*", default=[REFERENCE], help="Scripted AIs every network plays")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--games", type=int, default=200, help="Per pairing; even, each map from both seats")
    parser.add_argument("--first-map", type=int, default=FIRST_COMPARISON_MAP)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def describe(first: str, second: str, points: float, games: int, wins: int, draws: int) -> str:
    score = points / games if games else 0.0
    error = 0.5 / math.sqrt(games) if games else 1.0
    return (
        f"{first} vs {second}: {score:6.1%} ± {error:4.1%} of {games} "
        f"({wins} W {draws} D {games - wins - draws} L; {(score - 0.5) / error:+.1f} SE)"
    )


def main() -> None:
    import os

    os.chdir(ROOT)
    from game_types import BLUE, RED

    from ml.evaluation.evaluate import evaluation_maps
    from ml.judge.progress import ProgressBlock, bar, duration
    from ml.map_generator import generate_map
    from ml.models.checkpoint import load_checkpoint
    from ml.play.rollout import Matchup, RolloutPool, Seat
    from ml.training.league import fit_ratings

    args = parse_args()
    networks = dict(entry.split("=", 1) for entry in args.network)
    labels = list(networks)
    pairings = [(first, second) for index, first in enumerate(labels) for second in labels[index + 1 :]]
    pairings += [(network, opponent) for network in labels for opponent in args.opponents]

    maps = [
        generate_map(int(key)).definition
        for _, key in evaluation_maps(args.games // 2, bundled_maps=False, skip_generated=args.first_map)
    ]
    matchups, owners = [], []
    for first, second in pairings:
        for definition in maps:
            matchups += [Matchup(definition, Seat(first), Seat(second)), Matchup(definition, Seat(second), Seat(first))]
            owners += [(first, second, RED), (first, second, BLUE)]
    models = {label: load_checkpoint(path).model for label, path in networks.items()}
    temperatures = {label: args.temperature for label in labels}

    tally = {pairing: [0.0, 0, 0, 0] for pairing in pairings}  # points, games, wins, draws of the first
    block = ProgressBlock(
        f"Round robin '{args.name}': {len(labels)} networks at temperature {args.temperature:g}, "
        f"{', '.join(args.opponents) or 'no scripted AIs'}; {len(maps) * 2} games per pairing, from map {args.first_map}"
    )
    started, done = time.monotonic(), [0]

    def lines() -> list[str]:
        elapsed = time.monotonic() - started
        left = f", about {duration(elapsed / done[0] * (len(matchups) - done[0]))} left" if done[0] else ""
        rows = [f"{bar(done[0], len(matchups), 30)} {done[0]}/{len(matchups)} games, {duration(elapsed)}{left}"]
        for (first, second), (points, games, wins, draws) in tally.items():
            rows.append("  " + describe(first, second, points, games, wins, draws) if games else f"  {first} vs {second}: waiting")
        return rows

    def finished(index: int, _, winner: int | None) -> None:
        first, second, seat = owners[index]
        entry = tally[first, second]
        result = 0.5 if winner is None else float(winner == seat)
        entry[0] += result
        entry[1] += 1
        entry[2] += result == 1.0
        entry[3] += result == 0.5
        done[0] += 1
        block.show(lines())

    block.show(lines())
    with RolloutPool(args.workers) as pool:
        pool.play(matchups, models, seed=0, on_finished=finished, temperatures=temperatures)
    block.finish(lines())

    results = {pairing: [points, games] for pairing, (points, games, _, _) in tally.items()}
    anchors = (REFERENCE,) if REFERENCE in args.opponents else ()
    ratings = fit_ratings(results, anchors)
    print("\nRatings" + (f" ({REFERENCE} = 1000)" if anchors else "") + ":")
    for player, rating in sorted(ratings.items(), key=lambda item: -item[1]):
        print(f"  {player:<24} {rating:6.0f}")

    output = ROOT / "ml" / "runs" / "logs" / f"round_robin_{args.name}.json"
    output.write_text(json.dumps({
        "networks": networks, "opponents": args.opponents, "temperature": args.temperature,
        "first map": args.first_map, "games per pairing": len(maps) * 2,
        "results": [[first, second, points, games, wins, draws] for (first, second), (points, games, wins, draws) in tally.items()],
        "ratings": {player: round(rating) for player, rating in ratings.items()},
        "seconds": round(time.monotonic() - started),
    }, indent=1) + "\n")
    print(f"Written to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
