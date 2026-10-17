from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from ricochet.game.maps import available_maps, is_generated_map_id, save_map
from ricochet.runtime.loading import available_ais
from ricochet.runtime.match import MatchSummary, run_headless
from ricochet.runtime.replay import verify_replay
from ricochet.runtime.tournament import (
    head_to_head,
    run_generated_tournament,
    run_tournament,
    standings,
    write_results,
)
from ricochet.themes import DEFAULT_THEME, available_themes


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _map_id(value: str) -> str:
    if value in available_maps() or is_generated_map_id(value):
        return value
    raise argparse.ArgumentTypeError(
        f"unknown map {value!r}; choose from {', '.join(available_maps())} or generated:SEED"
    )


def _show_progress(done: int, total: int) -> None:
    end = "\n" if done == total else ""
    print(f"\r{done}/{total} games", end=end, file=sys.stderr, flush=True)


def _report(results: list[MatchSummary], output: Path, pairwise: bool = False) -> None:
    write_results(results, output)
    table = standings(results)
    print("bot                 score  wins  draws  losses")
    for bot, score, wins, draws, losses in table:
        print(f"{bot:20} {score:5.1f} {wins:5} {draws:6} {losses:7}")
    if pairwise:
        records = head_to_head(results)
        bots = [row[0] for row in table]
        print("\nHead-to-head score % (row against column)")
        print(" " * 23 + "".join(f"{index:>7}" for index in range(1, len(bots) + 1)))
        for index, bot in enumerate(bots, 1):
            cells = []
            for opponent in bots:
                wins, draws, losses = records.get((bot, opponent), (0, 0, 0))
                games = wins + draws + losses
                cells.append(f"{100 * (wins + draws / 2) / games:7.1f}" if games else "      -")
            print(f"{index:>2} {bot:20}" + "".join(cells))
    invalid = sum(result.invalid for result in results)
    if invalid:
        print(f"\n{invalid} invalid games (both controllers failed) are excluded.")
    print(f"\nWrote {len(results)} matches to {output} and {output.with_suffix('.csv')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ricochet human and AI arena")
    parser.add_argument("--theme", choices=available_themes(), default=DEFAULT_THEME)
    subparsers = parser.add_subparsers(dest="command")

    match = subparsers.add_parser("match", help="run one match")
    match.add_argument("--theme", choices=available_themes(), default=argparse.SUPPRESS)
    match.add_argument("--p1", default="direct_ai")
    match.add_argument("--p2", default="bank_shot_ai")
    match.add_argument(
        "--map", default="classic", type=_map_id, help="map file stem or generated:SEED"
    )
    match.add_argument("--seed", type=int, default=0)
    match.add_argument("--headless", action="store_true")
    match.add_argument("--record", type=Path)

    tournament = subparsers.add_parser("tournament", help="run a seat-swapped tournament")
    tournament.add_argument("--bots", nargs="+", default=["all"])
    tournament.add_argument("--maps", nargs="+", default=["all"])
    tournament.add_argument("--seeds", type=_csv_ints, default=(0,))
    tournament.add_argument("--output", type=Path, default=Path("results/tournament.json"))
    tournament.add_argument("--workers", type=_positive_int, default=1)

    auto = subparsers.add_parser(
        "auto-tournament", help="round robin on freshly generated maps, both seats per round"
    )
    auto.add_argument("--bots", nargs="+", default=["all"])
    auto.add_argument(
        "--rounds", type=_positive_int, default=100, help="generated maps per pairing"
    )
    auto.add_argument("--seed", type=int, default=0, help="round r uses map and match seed+r")
    auto.add_argument("--workers", type=_positive_int, help="parallel processes (default: CPUs)")
    auto.add_argument("--output", type=Path, default=Path("results/auto_tournament.json"))

    generate = subparsers.add_parser("generate-map", help="write one generated map as JSON")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--output", type=Path, help="file to write (default: print)")

    replay = subparsers.add_parser("replay", help="verify or display a replay")
    replay.add_argument("--theme", choices=available_themes(), default=argparse.SUPPRESS)
    replay.add_argument("path", type=Path)
    replay.add_argument("--headless", action="store_true")
    replay.add_argument("--speed", type=float, default=1.0)
    train = subparsers.add_parser("train", help="train the CPU PPO baseline")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--config", type=Path)
    train.add_argument("--seeds", type=_csv_ints)
    train.add_argument("--steps", type=int)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--resume", type=Path)
    evaluate = subparsers.add_parser("evaluate", help="evaluate a trusted checkpoint")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--suite", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--quick", action="store_true")
    evaluate.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    benchmark = subparsers.add_parser("benchmark", help="measure local simulation throughput")
    benchmark.add_argument("--steps", type=int, default=10000)
    balance = subparsers.add_parser("balance", help="compare shield presets")
    balance.add_argument("--output", type=Path, default=Path("results/shield_comparison.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        from ricochet.training.train import TrainConfig, train

        values = json.loads(args.config.read_text()) if args.config else {}
        seeds = args.seeds or values.pop("seeds", None)
        if args.seed is not None:
            values["seed"] = args.seed
        if args.steps is not None:
            values["total_timesteps"] = args.steps
        if args.smoke:
            values.update(total_timesteps=8192, smoke=True)
        if seeds:
            if args.seed is not None:
                raise ValueError("choose --seed or --seeds, not both")
            for seed in seeds:
                print(
                    train(
                        args.output / f"seed_{seed}",
                        TrainConfig(**(values | {"seed": seed})),
                        args.resume,
                    )
                )
        else:
            print(train(args.output, TrainConfig(**values), args.resume))
        return
    if args.command in ("evaluate", "benchmark", "balance"):
        from ricochet.training.evaluation import benchmark, compare_shields, evaluate

        if args.command == "benchmark":
            print(json.dumps(benchmark(args.steps), indent=2))
        elif args.command == "balance":
            print(f"Wrote {len(compare_shields(args.output))} matches to {args.output}")
        else:
            from ricochet.training.checkpoints import CheckpointController

            result = evaluate(CheckpointController(args.checkpoint), args.suite, args.quick)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "matches"}, indent=2))
        return
    if args.command is None:
        from ricochet.display.app import run_menu

        run_menu(args.theme)
        return
    if args.command == "match":
        if args.headless:
            if "human" in (args.p1, args.p2):
                raise ValueError("human players require the visual match")
            summary = run_headless(args.p1, args.p2, args.map, args.seed, record_path=args.record)
        else:
            if args.p1 == args.p2 == "human":
                raise ValueError("Ricochet supports at most one local human")
            from ricochet.display.app import run_visual_match
            from ricochet.display.menu import MenuSelection

            summary, _rematch = run_visual_match(
                MenuSelection(args.map, args.p1, args.p2, args.theme),
                args.seed,
                record_path=args.record,
            )
        print(json.dumps(summary.to_dict(), indent=2))
        return
    if args.command == "tournament":
        bots = available_ais() if args.bots == ["all"] else tuple(args.bots)
        maps = available_maps() if args.maps == ["all"] else tuple(args.maps)
        results = run_tournament(
            tuple(bots), tuple(maps), tuple(args.seeds), args.workers, _show_progress
        )
        _report(results, args.output)
        return
    if args.command == "auto-tournament":
        bots = available_ais() if args.bots == ["all"] else tuple(args.bots)
        results = run_generated_tournament(
            tuple(bots), args.rounds, args.seed, args.workers, on_result=_show_progress
        )
        _report(results, args.output, pairwise=True)
        return
    if args.command == "generate-map":
        from ricochet.game.mapgen import generate_map

        definition = generate_map(args.seed)
        if args.output is None:
            print(json.dumps(definition.to_dict(), indent=2))
        else:
            # Saved maps are loaded by file stem, so the file's ID follows its name.
            save_map(replace(definition, id=args.output.stem), args.output)
            print(f"Wrote {definition.name} to {args.output}")
        return
    if args.command == "replay":
        if args.headless:
            print(json.dumps(verify_replay(args.path), indent=2))
        else:
            from ricochet.display.app import run_visual_replay

            run_visual_replay(args.path, args.speed, args.theme)


if __name__ == "__main__":
    main()
