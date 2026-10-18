"""The ``ricochet`` command: the menu, single matches, replays, map generation and training."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from ricochet.game.maps import available_maps, is_generated_map_id, save_map
from ricochet.runtime.match import run_headless
from ricochet.runtime.replay import verify_replay
from ricochet.themes import DEFAULT_THEME, available_themes


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _map_id(value: str) -> str:
    if value in available_maps() or is_generated_map_id(value):
        return value
    raise argparse.ArgumentTypeError(
        f"unknown map {value!r}; choose from {', '.join(available_maps())} or generated:SEED"
    )


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
    if args.command == "evaluate":
        from ricochet.training.checkpoints import CheckpointController
        from ricochet.training.evaluation import evaluate

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
