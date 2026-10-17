"""Compare training settings by running them and letting the results play each other.

A sweep trains the same starting model several times: once per variation of
the settings, and each variation with several seeds. Afterwards every
resulting model plays the starting model, the scripted AIs, and the models
from the other variations, on maps no trainer has seen. All of those matches
together give each model one rating.

The seeds are the point. Two runs with identical settings differ by chance
alone; how much they differ is the smallest effect a sweep can show. A
variation only counts as better if it beats that.

Run from the game directory:

    .venv/bin/python -m ml sweep --name entropy --trainer selfplay \\
        --init ml/runs/imitation_v1/best.pt --seeds 0 1 \\
        --set rounds=20 --vary entropy_weight=0,0.003,0.01

``--set`` changes a setting for every run. ``--vary`` tries each listed value
of one setting, everything else staying as it is; it can be given more than
once. ``--variant`` adds one variation that changes several settings together. Runs that already finished are not repeated, so an interrupted sweep
can simply be started again. Results land in ``ml/runs/sweep_<name>/``.
"""

import argparse
import json
import statistics
import subprocess
import sys
import traceback
from dataclasses import asdict, fields, replace

from game_types import BLUE, RED

import ml.training.runlog
from ml.evaluation.evaluate import DEFAULT_OPPONENTS, evaluation_maps
from ml.map_generator import generate_map
from ml.models.checkpoint import init_model, load_checkpoint
from ml.play.rollout import Matchup, RolloutPool, Seat
from ml.training.league import fit_ratings
from ml.training.runlog import read_metrics
from ml.training.train import TRAINERS, yes_or_no


START = "starting model"
# Sweeps are judged on generated evaluation maps from this position on. The
# maps before it are used by trainers and for final verdicts.
FIRST_TUNING_MAP = 100


def run_sweep(
    name: str,
    trainer: str,
    init: str,
    fixed: dict,
    variations: list[dict],
    seeds: list[int],
    match_games: int = 200,
    workers: int = 6,
    separate_processes: bool = False,
) -> dict:
    """Train every variation with every seed, then rate the results. Returns the report as data.

    With ``separate_processes``, every run trains in a fresh Python process,
    so whatever memory one run leaves behind cannot slow down the next.
    """

    settings_class, train = TRAINERS[trainer]
    base = replace(settings_class(), **fixed)
    folder = ml.training.runlog.RUNS_DIRECTORY / f"sweep_{name}"

    runs = []
    tried = []
    for changes in variations:
        label = ", ".join(f"{key}={value}" for key, value in changes.items()) or "as given"
        if replace(base, **changes) in tried:
            # The same settings under another name, such as a default value listed twice.
            continue
        tried.append(replace(base, **changes))
        for seed in seeds:
            settings = replace(base, **changes, seed=seed)
            run_name = f"sweep_{name}/{_file_name(label)}_seed{seed}"
            runs.append({"variation": label, "seed": seed, "run": run_name, "settings": settings})

    for number, run in enumerate(runs, start=1):
        if _finished(run["run"], run["settings"]):
            print(f"=== {number}/{len(runs)}: {run['run']} is already done", flush=True)
            continue
        print(f"=== {number}/{len(runs)}: training {run['run']}", flush=True)
        try:
            if separate_processes:
                subprocess.run(_command(trainer, run["run"], init, run["settings"]), check=True)
            else:
                train(run["run"], run["settings"], init=init, overwrite=True)
        except Exception:
            # One failed run should not take the rest of the sweep with it.
            print(traceback.format_exc(), flush=True)

    finished = [run for run in runs if _finished(run["run"], run["settings"])]
    print(f"=== rating {len(finished)} finished runs", flush=True)
    report = rate_runs(finished, init, match_games, workers)
    report.update(name=name, trainer=trainer, init=str(init), fixed=fixed, seeds=seeds)

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    text = describe(report)
    (folder / "report.txt").write_text(text + "\n")
    print("\n" + text, flush=True)
    return report


def rate_runs(runs: list[dict], init: str, match_games: int, workers: int) -> dict:
    """Play the matches and work out one rating per run."""

    models = {run["run"]: load_checkpoint(ml.training.runlog.RUNS_DIRECTORY / run["run"] / "latest.pt").model for run in runs}
    models[START] = init_model(init, seed=0).model
    maps = [
        generate_map(int(key)).definition
        for _, key in evaluation_maps(match_games // 2, bundled_maps=False, skip_generated=FIRST_TUNING_MAP)
    ]

    # Every run meets the starting model and each scripted AI on all the maps
    # from both seats. Runs from different variations meet on a quarter of them.
    pairings = [(run["run"], rival, maps) for run in runs for rival in (START, *DEFAULT_OPPONENTS)]
    for index, first in enumerate(runs):
        for second in runs[index + 1 :]:
            if first["variation"] != second["variation"]:
                pairings.append((first["run"], second["run"], maps[: max(1, len(maps) // 4)]))

    matchups, owners = [], []
    for first, second, on_maps in pairings:
        for definition in on_maps:
            matchups += [Matchup(definition, Seat(first), Seat(second)), Matchup(definition, Seat(second), Seat(first))]
            owners += [(first, second, RED), (first, second, BLUE)]

    results: dict[tuple[str, str], list[float]] = {}
    with RolloutPool(workers) as pool:
        records = pool.play(matchups, models, seed=0)
    for (first, second, seat), record in zip(owners, records):
        entry = results.setdefault((first, second), [0.0, 0])
        entry[0] += 0.5 if record.winner is None else float(record.winner == seat)
        entry[1] += 1

    ratings = fit_ratings(results, DEFAULT_OPPONENTS)

    def score(first: str, second: str) -> float:
        points, games = results[(first, second)]
        return points / games

    rows = []
    for run in runs:
        rounds = read_metrics(run["run"])
        final = rounds[-1]
        rows.append(
            {
                "variation": run["variation"],
                "seed": run["seed"],
                "run": run["run"],
                "rating": ratings[run["run"]],
                "against the starting model": score(run["run"], START),
                **{f"against {opponent}": score(run["run"], opponent) for opponent in DEFAULT_OPPONENTS},
                "entropy at the end": final.get("entropy/of the policy"),
                "minutes": final["seconds"] / 60,
                "settings": asdict(run["settings"]),
            }
        )
    return {
        "runs": rows,
        "ratings": {name: rating for name, rating in ratings.items() if name not in models or name == START},
        "games per match": len(maps) * 2,
        "results": [[first, second, points, games] for (first, second), (points, games) in results.items()],
    }


def describe(report: dict) -> str:
    """The report as a table: one line per variation, averaged over its seeds."""

    by_variation: dict[str, list[dict]] = {}
    for row in report["runs"]:
        by_variation.setdefault(row["variation"], []).append(row)

    lines = [
        f"Sweep '{report['name']}': {report['trainer']} from {report['init']}, "
        f"{report['games per match']} games per match",
        "",
        f"{'variation':34}{'runs':>5}{'rating':>8}{'lowest-highest':>17}"
        f"{'vs start':>10}{'vs strategic':>14}{'vs careful':>12}{'vs aggressive':>15}{'entropy':>9}{'minutes':>9}",
    ]
    ordered = sorted(by_variation.items(), key=lambda item: -statistics.mean(row["rating"] for row in item[1]))
    for variation, rows in ordered:
        ratings = [row["rating"] for row in rows]

        def mean(key):
            values = [row[key] for row in rows if row[key] is not None]
            return statistics.mean(values) if values else float("nan")

        lines.append(
            f"{variation:34}{len(rows):5d}{statistics.mean(ratings):8.0f}"
            f"{f'{min(ratings):.0f}-{max(ratings):.0f}':>17}"
            f"{mean('against the starting model'):10.1%}{mean('against strategic_ai'):14.1%}"
            f"{mean('against careful_ai'):12.1%}{mean('against aggressive_ai'):15.1%}"
            f"{mean('entropy at the end'):9.2f}{mean('minutes'):9.1f}"
        )
    reference = report["ratings"]
    lines += [
        "",
        "For reference: "
        + ", ".join(f"{name} {rating:.0f}" for name, rating in sorted(reference.items(), key=lambda item: -item[1])),
        "A variation is only better than another if the gap between them is clearly larger than",
        "the gap between the lowest and highest run of the same variation.",
    ]
    return "\n".join(lines)


def _command(trainer: str, run_name: str, init: str, settings) -> list[str]:
    """The ``ml.training.train`` command line that trains one run with exactly these settings."""

    command = [sys.executable, "-u", "-m", "ml", "train", trainer, "--run", run_name, "--init", str(init), "--overwrite"]
    for setting in fields(settings):
        value = getattr(settings, setting.name)
        option = "--" + setting.name.replace("_", "-")
        if isinstance(value, tuple):
            command += [option, *value]
        elif isinstance(value, bool):
            command += [option, "yes" if value else "no"]
        else:
            command += [option, repr(value) if isinstance(value, float) else str(value)]
    return command


def _finished(run_name: str, settings) -> bool:
    """True if the run's log reaches its last round (or epoch, for imitation)."""

    try:
        logged = read_metrics(run_name)
    except FileNotFoundError:
        return False
    last = getattr(settings, "rounds", None) or settings.epochs
    return bool(logged) and logged[-1]["round"] >= last


def _file_name(label: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in label)


def _convert(settings_class, name: str, text: str):
    """Turn text from the command line into the type the setting has."""

    defaults = {setting.name: setting.default for setting in fields(settings_class)}
    if name not in defaults:
        raise SystemExit(f"'{name}' is not a setting of this trainer. Settings: {', '.join(defaults)}")
    default = defaults[name]
    if isinstance(default, bool):
        return yes_or_no(text)
    if isinstance(default, tuple):
        return tuple(text.split("+"))
    return type(default)(text)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="The sweep is written to ml/runs/sweep_NAME/")
    parser.add_argument("--trainer", default="selfplay", choices=sorted(TRAINERS))
    parser.add_argument("--init", default="random", help='"random" or the checkpoint every run starts from')
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--set", nargs="*", default=[], metavar="SETTING=VALUE", help="Applied to every run")
    parser.add_argument(
        "--vary",
        action="append",
        default=[],
        metavar="SETTING=V1,V2,...",
        help="Try each value, one at a time. Separate the parts of a list value with +",
    )
    parser.add_argument(
        "--variant",
        action="append",
        nargs="+",
        default=[],
        metavar="SETTING=VALUE",
        help="One more variation, changing all the listed settings together",
    )
    parser.add_argument("--match-games", type=int, default=200, help="Games per match when rating the results")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--same-process",
        action="store_true",
        help="Train every run inside this one process instead of a fresh process each",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings_class, _ = TRAINERS[args.trainer]

    fixed = {}
    for item in args.set:
        name, _, text = item.partition("=")
        fixed[name] = _convert(settings_class, name, text)

    variations = []
    for item in args.vary:
        name, _, texts = item.partition("=")
        for text in texts.split(","):
            changes = {name: _convert(settings_class, name, text)}
            if changes not in variations:
                variations.append(changes)
    for items in args.variant:
        changes = {}
        for item in items:
            name, _, text = item.partition("=")
            changes[name] = _convert(settings_class, name, text)
        variations.append(changes)
    if not variations:
        variations = [{}]

    run_sweep(
        args.name, args.trainer, args.init, fixed, variations, args.seeds,
        args.match_games, args.workers, separate_processes=not args.same_process,
    )


if __name__ == "__main__":
    main()
