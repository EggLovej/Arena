"""Put checkpoints from several training runs on one rating scale.

Each chosen checkpoint (say rounds 0, 25, 50, 100, 150 of every run) plays a
fixed panel of opponents on fresh maps, and the last checkpoints of runs from
different groups also play each other directly. All results are rated
together, with ``strategic_ai`` fixed at 1000, so curves from different runs
can be compared point by point.

Runs are named ``GROUP/LABEL=PATH``, for example ``random start/seed 1=ml/runs/x``.
Groups are what gets compared; each group's runs are its seeds.

A run trained under a send limit (``--min-send-gap``) plays under it here
too, every round including round 0, and so does a panel checkpoint from
such a run; every other player has no limit. ``--unlimited`` lifts every
limit, to see how such networks play once it is gone.

Time is counted from the very beginning: for a run that started from an
imitation checkpoint, the imitation run's own time is added (found through
the checkpoint's history).

    .venv/bin/python -m ml compare --name stage10 \\
        --run "random start/seed 0=ml/runs/overnight_scratch" \\
        --run "imitation start/seed 0=ml/runs/stage10_imitation_start_seed0" \\
        --rounds 0 50 100 150 --panel strategic_ai ml/runs/overnight_scratch/round_0300.pt

Results land in ``ml/runs/compare_NAME/``: a table, the raw numbers, and a
``metrics.jsonl`` the monitor can draw (``--run compare_NAME``).
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from game_types import BLUE, RED

import ml.training.runlog
from ml.evaluation.evaluate import evaluation_maps
from ml.judge.progress import ProgressBlock, bar, duration
from ml.map_generator import generate_map
from ml.models.checkpoint import load_checkpoint
from ml.play.players import is_network_player, min_send_gap
from ml.play.rollout import Matchup, RolloutPool, Seat
from ml.training.league import fit_ratings
from ml.training.runlog import read_metrics


REFERENCE = "strategic_ai"
# Fresh maps, past the ones trainers, evaluations, and tuning sweeps have used.
FIRST_COMPARISON_MAP = 300


@dataclass(frozen=True)
class Checkpoint:
    group: str
    label: str
    round: int
    path: Path
    hours: float  # training time from the very start, imitation included

    @property
    def name(self) -> str:
        return f"{self.group}/{self.label}/round {self.round}"


def compare(
    name: str,
    runs: dict[tuple[str, str], Path],
    rounds: list[int],
    panel: list[str],
    games: int = 100,
    workers: int = 6,
    inference_device: str | None = None,
    unlimited: bool = False,
) -> dict:
    checkpoints = [
        _checkpoint(group, label, folder, round_number)
        for (group, label), folder in runs.items()
        for round_number in rounds
        if (Path(folder) / f"round_{round_number:04d}.pt").exists()
    ]

    models = {checkpoint.name: load_checkpoint(checkpoint.path).model for checkpoint in checkpoints}
    for entry in panel:
        if _is_checkpoint(entry):
            models[entry] = load_checkpoint(entry).model

    gaps = {checkpoint.name: min_send_gap(checkpoint.path) for checkpoint in checkpoints}
    gaps |= {entry: min_send_gap(entry) for entry in panel if _is_checkpoint(entry)}
    gaps = {player: gap for player, gap in gaps.items() if gap > 1 and not unlimited}

    pairings = [(checkpoint.name, opponent) for checkpoint in checkpoints for opponent in panel]
    # The last checkpoints of runs from different groups also meet head to head.
    last = {}
    for checkpoint in checkpoints:
        key = (checkpoint.group, checkpoint.label)
        if key not in last or checkpoint.round > last[key].round:
            last[key] = checkpoint
    finals = list(last.values())
    pairings += [
        (first.name, second.name)
        for index, first in enumerate(finals)
        for second in finals[index + 1 :]
        if first.group != second.group
    ]

    maps = [
        generate_map(int(key)).definition
        for _, key in evaluation_maps(games // 2, bundled_maps=False, skip_generated=FIRST_COMPARISON_MAP)
    ]
    matchups, owners = [], []
    for first, second in pairings:
        for definition in maps:
            one, other = (Seat(player, min_send_gap=gaps.get(player, 0)) for player in (first, second))
            matchups += [Matchup(definition, one, other), Matchup(definition, other, one)]
            owners += [(first, second, RED), (first, second, BLUE)]
    print(f"Playing {len(matchups):,} games for {len(checkpoints)} checkpoints", flush=True)
    if gaps:
        print(_describe_gaps(gaps), flush=True)
    live = _LiveBoard(pairings, len(maps) * 2)
    # With a device, the networks are served from it in batches (ml/play/inference.py): worth it on
    # an NVIDIA GPU, not on the Apple one.
    with RolloutPool(workers, inference_device=inference_device) as pool:
        records = pool.play(
            matchups, models, seed=0,
            on_finished=lambda index, _, winner: live.add(*owners[index], winner),
        )
    live.done()

    results: dict[tuple[str, str], list[float]] = {}
    for (first, second, seat), record in zip(owners, records):
        entry = results.setdefault((first, second), [0.0, 0])
        entry[0] += 0.5 if record.winner is None else float(record.winner == seat)
        entry[1] += 1
    ratings = fit_ratings(results, (REFERENCE,))

    rows = [
        {
            "group": checkpoint.group,
            "label": checkpoint.label,
            "round": checkpoint.round,
            "hours": checkpoint.hours,
            "rating": ratings[checkpoint.name],
            f"against {REFERENCE}": _score(results, checkpoint.name, REFERENCE),
        }
        for checkpoint in checkpoints
    ]
    head_to_head = [
        {"first": first, "second": second, "score": points / count, "games": count}
        for (first, second), (points, count) in results.items()
        if first in {c.name for c in finals} and second in {c.name for c in finals}
    ]
    report = {
        "name": name,
        # Which run folder each group/label is, so ml lineage can find the ratings of a run.
        "runs": {f"{group}/{label}": str(folder) for (group, label), folder in runs.items()},
        "rounds": rounds,
        "panel": panel,
        "send gaps": gaps,
        "unlimited": unlimited,
        "games per match": len(maps) * 2,
        "checkpoints": rows,
        "head to head": head_to_head,
        "panel ratings": {entry: ratings[entry] for entry in panel},
        "results": [[first, second, points, count] for (first, second), (points, count) in results.items()],
    }

    folder = ml.training.runlog.RUNS_DIRECTORY / f"compare_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    text = describe(report)
    (folder / "report.txt").write_text(text + "\n")
    _write_chart_data(folder, rows)
    print("\n" + text, flush=True)
    return report


class _LiveBoard:
    """Wins, draws, and losses of every match so far, one line per match, updated after every game."""

    def __init__(self, pairings: list[tuple[str, str]], games_per_match: int):
        self.pairings = pairings
        self.games_per_match = games_per_match
        self.tally = {pairing: [0, 0, 0] for pairing in pairings}  # wins, draws, losses of the first
        self.started = time.monotonic()
        self.block = ProgressBlock("Matches (score of the left side)")
        self.block.show(self.lines())

    def add(self, first: str, second: str, seat: int, winner: int | None) -> None:
        tally = self.tally[(first, second)]
        tally[1 if winner is None else 0 if winner == seat else 2] += 1
        self.block.show(self.lines())

    def lines(self) -> list[str]:
        width = max(len(first) for first, _ in self.pairings)
        opponent_width = max(len(_short(second)) for _, second in self.pairings)
        lines = []
        for (first, second), (wins, draws, losses) in self.tally.items():
            played = wins + draws + losses
            score = f"{(wins + draws / 2) / played:6.1%}" if played else "      "
            lines.append(
                f"{first:<{width}}  vs {_short(second):<{opponent_width}}  {bar(played, self.games_per_match, 10)} "
                f"{played:3d}/{self.games_per_match}  {wins:3d} W {draws:2d} D {losses:3d} L  {score}"
            )
        done = sum(sum(tally) for tally in self.tally.values())
        total = len(self.pairings) * self.games_per_match
        spent = time.monotonic() - self.started
        left = f", about {duration(spent / done * (total - done))} left" if done else ""
        lines.append(f"{done:,}/{total:,} games, {duration(spent)} so far{left}")
        return lines

    def done(self) -> None:
        self.block.finish(self.lines())


def _short(player: str) -> str:
    """A panel checkpoint by its run and round, e.g. deep_scratch/round_0300."""

    path = Path(player)
    return f"{path.parent.name}/{path.stem}" if _is_checkpoint(player) else player


def _is_checkpoint(player: str) -> bool:
    """A panel entry that is a checkpoint, not a player built on one (``lookahead:...?judge=x.pt``)."""

    return player.endswith(".pt") and not is_network_player(player)


def describe(report: dict) -> str:
    groups = sorted({row["group"] for row in report["checkpoints"]})
    lines = [
        f"Comparison '{report['name']}': {report['games per match']} games per match, "
        f"{REFERENCE} fixed at 1000",
        "",
        "Rating by round (mean over seeds, lowest-highest in brackets), and hours of training to get there:",
        f"{'round':>7}" + "".join(f"{group:>36}" for group in groups),
    ]
    for round_number in report["rounds"]:
        cells = []
        for group in groups:
            rows = [r for r in report["checkpoints"] if r["group"] == group and r["round"] == round_number]
            if not rows:
                cells.append(f"{'-':>36}")
                continue
            ratings = [r["rating"] for r in rows]
            hours = statistics.mean(r["hours"] for r in rows)
            cell = f"{statistics.mean(ratings):.0f} ({min(ratings):.0f}-{max(ratings):.0f}) at {hours:.1f} h"
            cells.append(f"{cell:>36}")
        lines.append(f"{round_number:>7}" + "".join(cells))

    lines += ["", f"Score against {REFERENCE} by round (mean over seeds):", f"{'round':>7}" + "".join(f"{g:>36}" for g in groups)]
    for round_number in report["rounds"]:
        cells = []
        for group in groups:
            rows = [r for r in report["checkpoints"] if r["group"] == group and r["round"] == round_number]
            value = f"{statistics.mean(r[f'against {REFERENCE}'] for r in rows):.0%}" if rows else "-"
            cells.append(f"{value:>36}")
        lines.append(f"{round_number:>7}" + "".join(cells))

    pooled: dict[tuple[str, str], list[float]] = {}
    group_of = {f"{r['group']}/{r['label']}/round {r['round']}": r["group"] for r in report["checkpoints"]}
    for match in report["head to head"]:
        key = (group_of[match["first"]], group_of[match["second"]])
        entry = pooled.setdefault(key, [0.0, 0])
        entry[0] += match["score"] * match["games"]
        entry[1] += match["games"]
    if pooled:
        lines += ["", "Last checkpoints head to head (all seeds pooled):"]
        for (first, second), (points, games) in pooled.items():
            spread = (0.25 / games) ** 0.5
            lines.append(
                f"  {first} against {second}: {points / games:.1%} of {games} games "
                f"({(points / games - 0.5) / spread:+.1f} standard errors from even)"
            )
    lines += ["", "Panel: " + ", ".join(f"{entry} {rating:.0f}" for entry, rating in report["panel ratings"].items())]
    if report.get("send gaps"):
        lines += ["", _describe_gaps(report["send gaps"])]
    if report.get("unlimited"):
        lines += ["", "Every network played without a send limit (--unlimited), whatever it trained under."]
    return "\n".join(lines)


def _describe_gaps(gaps: dict[str, int]) -> str:
    """Which players were limited to one send every so many ticks (everyone else: no limit)."""

    return "Limited to one send every N ticks, as trained: " + ", ".join(
        f"{_short(player)} {gap}" for player, gap in gaps.items()
    )


def _checkpoint(group: str, label: str, folder: Path, round_number: int) -> Checkpoint:
    folder = Path(folder)
    logged = {entry["round"]: entry["seconds"] for entry in read_metrics(folder)}
    seconds = logged.get(round_number, 0.0) + _head_start_seconds(folder)
    return Checkpoint(group, label, round_number, folder / f"round_{round_number:04d}.pt", seconds / 3600)


def _head_start_seconds(folder: Path) -> float:
    """Training time spent before this run started, such as the imitation it began from."""

    history = load_checkpoint(folder / "round_0000.pt").history
    total = 0.0
    for entry in history:
        run = entry.get("run")
        if run and run != folder.name:
            earlier = ml.training.runlog.RUNS_DIRECTORY / run / "metrics.jsonl"
            if earlier.exists():
                logged = read_metrics(earlier.parent)
                total += logged[-1]["seconds"] if logged else 0.0
    return total


def _score(results: dict, first: str, second: str) -> float:
    points, games = results[(first, second)]
    return points / games


def _write_chart_data(folder: Path, rows: list[dict]) -> None:
    """One line per round, so the monitor can draw rating against round for each group."""

    lines = []
    for round_number in sorted({row["round"] for row in rows}):
        entry = {"round": round_number, "seconds": 0.0}
        for group in sorted({row["group"] for row in rows}):
            chosen = [row for row in rows if row["group"] == group and row["round"] == round_number]
            if chosen:
                entry[f"rating/{group}"] = statistics.mean(row["rating"] for row in chosen)
                entry[f"score/{group} vs {REFERENCE}"] = statistics.mean(row[f"against {REFERENCE}"] for row in chosen)
                entry[f"hours of training/{group}"] = statistics.mean(row["hours"] for row in chosen)
        lines.append(json.dumps(entry))
    (folder / "metrics.jsonl").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--run", action="append", required=True, metavar="GROUP/LABEL=PATH")
    parser.add_argument("--rounds", type=int, nargs="+", required=True)
    parser.add_argument("--panel", nargs="+", default=[REFERENCE], help="AI names and checkpoint paths every checkpoint plays")
    parser.add_argument("--games", type=int, default=100, help="Games per match")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gpu-server", action="store_true", help="Serve the networks from the GPU (pays off on an NVIDIA GPU)")
    parser.add_argument("--unlimited", action="store_true", help="No send limit for anyone, even runs trained under one")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = {}
    for item in args.run:
        key, _, path = item.partition("=")
        group, _, label = key.partition("/")
        runs[(group, label or path)] = Path(path)
    if REFERENCE not in args.panel:
        args.panel.insert(0, REFERENCE)
    device = None
    if args.gpu_server:
        from ml.device import default_device

        device = default_device()
    compare(args.name, runs, args.rounds, args.panel, args.games, args.workers, device, args.unlimited)


if __name__ == "__main__":
    main()
