"""A network against Beast, with the score updated live as the games come in.

The network draws its moves at a chosen temperature (Neural AI plays at 0.2; every comparison so
far at 1, by chance) and has no send limit. ``--player`` plays any other player instead, by the
name the evaluator understands (e.g. ``lookahead:PATH?judge=PATH``), with a ``--label`` for it. Beast is whoever ml/runs/beast.json names. The maps are
the comparisons' fresh ones (`ml compare`), both seats each, so by default the result pairs up with
compare_free_beast_check (the same network at temperature 1: 48.8%). ``--first-map`` starts further
on, for another batch on new maps. At the end the batch is written to
ml/runs/logs/beast_match_<label>_maps<first map>.json (the label of a network:
<run>_r<round>_t<temperature>), and all batches of the same label and Beast are added up. Clearly ahead: at least +2.0 standard errors
from even (57.1% over 200 games, 55.0% over 400).

Run from the game directory (about 25 minutes for 200 games):

    .venv/bin/python ml/scripts/beast_match.py
    .venv/bin/python ml/scripts/beast_match.py --first-map 400      the next 100 maps
    .venv/bin/python ml/scripts/beast_match.py --network ml/runs/free_gap0/round_0050.pt --temperature 0.2 --games 200
    .venv/bin/python ml/scripts/beast_match.py --player "lookahead:ml/runs/free_gap0/round_0050.pt?rule=nash" --label lookahead_own
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Standard errors from even that count as clearly ahead, enough to replace Beast.
CLEARLY = 2.0
FIRST_COMPARISON_MAP = 300  # as in ml/evaluation/compare.py


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--network", default="ml/runs/free_gap0/round_0050.pt")
    parser.add_argument("--temperature", type=float, default=0.2, help="1 draws by the network's own odds")
    parser.add_argument("--player", help="Instead of --network: any player name the evaluator understands")
    parser.add_argument("--label", help="A short name for the title and the file (required with --player)")
    parser.add_argument("--games", type=int, default=200, help="An even number: each map is played from both seats")
    parser.add_argument(
        "--first-map", type=int, default=FIRST_COMPARISON_MAP,
        help=f"Generated maps skipped before the first one played (default {FIRST_COMPARISON_MAP}, the comparisons' first)",
    )
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def score_line(label: str, results: list[float]) -> str:
    """Score, wins/draws/losses, and the standard error of the score."""

    games = len(results)
    if not games:
        return f"{label:8s} no games yet"
    score = sum(results) / games
    wins, draws = results.count(1.0), results.count(0.5)
    error = 0.5 / math.sqrt(games)  # of a score near 50%
    return (
        f"{label:8s} {score:6.1%} ± {error:4.1%}  ({wins} W {draws} D {games - wins - draws} L; "
        f"{(score - 0.5) / error:+.1f} SE from even)"
    )


def verdict(score: float, games: int) -> str:
    standard_errors = (score - 0.5) / (0.5 / math.sqrt(games))
    if standard_errors >= CLEARLY:
        return f"clearly ahead (+{CLEARLY:g} SE or more): it should become Beast"
    if score > 0.5:
        return f"ahead, but under +{CLEARLY:g} SE: not clearly; Beast stays"
    return "not ahead: Beast stays"


def all_batches(pattern: str, beast: str) -> None:
    """Add up every batch of this network, temperature and Beast."""

    batches = [json.loads(path.read_text()) for path in sorted((ROOT / "ml" / "runs" / "logs").glob(pattern))]
    batches = [batch for batch in batches if batch["beast"] == beast]
    if len(batches) < 2:
        return
    games = [game for batch in batches for game in batch["each game"]]
    if len({(game["map"], game["seat"]) for game in games}) < len(games):
        print("Batches overlap (a map and seat played twice); not added up.")
        return
    results = [game["result"] for game in games]
    score = sum(results) / len(results)
    ranges = ", ".join(f"maps {min(g['map'] for g in b['each game'])}-{max(g['map'] for g in b['each game'])}" for b in batches)
    print(f"\nAll {len(batches)} batches ({ranges}):")
    print(score_line("overall", results))
    print(verdict(score, len(results)))


def main() -> None:
    import os

    os.chdir(ROOT)
    from game_types import BLUE, RED

    from ml.evaluation.evaluate import evaluation_maps
    from ml.judge.progress import ProgressBlock, bar, duration
    from ml.map_generator import generate_map
    from ml.models.checkpoint import load_checkpoint
    from ml.play.beast import current
    from ml.play.rollout import Matchup, RolloutPool, Seat

    args = parse_args()
    beast = current()["player"]
    if args.player:
        if not args.label:
            raise SystemExit("--player needs a --label")
        # Played by name in the workers, as Beast is; no temperature.
        name, label, models, temperatures = args.player, args.label, {}, {}
        described = f"{args.label} ({args.player})"
    else:
        network = Path(args.network)
        round_number = int(network.stem.split("_")[1])
        name = f"{network.parent.name} r{round_number}"
        label = args.label or f"{network.parent.name}_r{round_number}_t{args.temperature:g}"
        models, temperatures = {name: load_checkpoint(network).model}, {name: args.temperature}
        described = f"{name} at temperature {args.temperature:g}"
    title = f"{described} against Beast ({beast}), from map {args.first_map}"

    maps = [
        (key, generate_map(int(key)).definition)
        for _, key in evaluation_maps(args.games // 2, bundled_maps=False, skip_generated=args.first_map)
    ]
    ours, theirs = Seat(name), Seat(beast)
    matchups, seats, keys = [], [], []
    for key, definition in maps:
        matchups += [Matchup(definition, ours, theirs), Matchup(definition, theirs, ours)]
        seats += [RED, BLUE]
        keys += [key, key]

    block = ProgressBlock(title)
    results = {RED: [], BLUE: []}
    games = []
    started = time.monotonic()

    def lines() -> list[str]:
        done = len(games)
        elapsed = time.monotonic() - started
        left = f", about {duration(elapsed / done * (len(matchups) - done))} left" if done else ""
        return [
            f"{bar(done, len(matchups), 30)} {done}/{len(matchups)} games, {duration(elapsed)}{left}",
            score_line("overall", results[RED] + results[BLUE]),
            score_line("as red", results[RED]),
            score_line("as blue", results[BLUE]),
        ]

    def finished(index: int, _, winner: int | None) -> None:
        seat = seats[index]
        result = 0.5 if winner is None else float(winner == seat)
        results[seat].append(result)
        games.append({"map": keys[index], "seat": "red" if seat == RED else "blue", "result": result})
        block.show(lines())

    block.show(lines())
    with RolloutPool(args.workers) as pool:
        pool.play(matchups, models, seed=0, on_finished=finished, temperatures=temperatures)
    block.finish(lines())

    everything = results[RED] + results[BLUE]
    score = sum(everything) / len(everything)
    this_batch = verdict(score, len(everything))
    print(f"\nThis batch: {this_batch}")

    stem = f"beast_match_{label}"
    output = ROOT / "ml" / "runs" / "logs" / f"{stem}_maps{args.first_map}.json"
    output.write_text(json.dumps({
        **({"player": args.player} if args.player else {"network": str(network), "temperature": args.temperature}),
        "beast": beast, "first map": args.first_map,
        "score": score, "games": len(everything), "verdict": this_batch, "seconds": round(time.monotonic() - started),
        "each game": games,
    }, indent=1) + "\n")
    print(f"Written to {output.relative_to(ROOT)}")
    all_batches(f"{stem}_maps*.json", beast)


if __name__ == "__main__":
    main()
