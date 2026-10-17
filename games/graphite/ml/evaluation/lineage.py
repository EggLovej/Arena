"""One set of training curves for a network trained over several runs, round after round.

A checkpoint's history lists every run that trained it (for the current
best: imitation, then deep_from_imitation, ab_plain, shaping_entropy). This
command follows that history and joins the runs' logged rounds into one
run folder, numbered from the first self-play round on, so ``ml monitor``
shows the whole road at once:

    .venv/bin/python -m ml lineage ml/runs/shaping_entropy/round_0150.pt
    .venv/bin/python -m ml monitor --run lineage_shaping_entropy

What is joined, and how:
- Every metric the runs logged, at its overall round. "score/vs starting
  model" becomes "score/vs the start of its run": the starting model is a
  different one in each run.
- The entropy bonus of each run, as its own chart, so it is visible where
  the settings changed.
- The ratings from ``ml compare`` comparisons that rated the runs' rounds
  (100 games per match, far less noisy than the 16-game checks during
  training). Each comparison has its own panel, so its scale is a little
  different; the ratings are chain-linked: each run's are shifted so its
  round 0 equals the previous run's last rated round, which is the same
  network. Comparisons record which runs they rated (since 2026-10-04);
  older ones are named with ``--ratings RUN=COMPARISON/GROUP``.

The joined folder is ``ml/runs/lineage_NAME/`` (``metrics.jsonl`` and a
``segments.json`` describing the parts).
"""

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from ml.models.checkpoint import load_checkpoint
from ml.training.runlog import RUNS_DIRECTORY, format_duration, read_metrics


# Ratings of runs rated before comparisons recorded their runs: run -> (comparison, group).
KNOWN_RATINGS = {
    "deep_from_imitation": ("stage10_deep", "imitation start"),
    "ab_plain": ("plain_400", "plain"),
    "shaping_entropy": ("shaping", "entropy 0.001"),
}


# Metrics logged under an older name.
RENAMED = {"behaviour/sends per game and seat": "sends/per game and seat"}


@dataclass
class Segment:
    run: str
    step: str
    rounds: int  # how many of the run's rounds went into the network
    first: int = 0  # the overall round of the run's round 0

    @property
    def last(self) -> int:
        return self.first + self.rounds


def segments_of(checkpoint: str) -> tuple[list[Segment], list[dict]]:
    """The runs that trained the network round by round, in order, and the other steps of its history."""

    history = load_checkpoint(checkpoint).history
    segments, other = [], []
    first = 0
    for entry in history:
        if "run" in entry and "rounds" in entry:
            segments.append(Segment(entry["run"], entry["step"], entry["rounds"], first))
            first += entry["rounds"]
        else:
            other.append(entry)
    return segments, other


def comparison_ratings(run: str, named: dict[str, tuple[str, str]]) -> dict[int, dict[str, float]]:
    """The run's rated rounds: round -> rating and score against strategic_ai, from one comparison."""

    comparison = group = None
    if run in named:
        comparison, group = named[run]
    else:
        for path in sorted(RUNS_DIRECTORY.glob("compare_*/results.json")):
            runs = json.loads(path.read_text()).get("runs", {})
            for key, folder in runs.items():
                if Path(folder).name == run:
                    comparison, group = path.parent.name.removeprefix("compare_"), key.split("/")[0]
    if comparison is None:
        return {}
    results = json.loads((RUNS_DIRECTORY / f"compare_{comparison}" / "results.json").read_text())
    rated: dict[int, dict[str, list[float]]] = {}
    for row in results["checkpoints"]:
        if row["group"] == group:
            entry = rated.setdefault(row["round"], {"rating": [], "score": []})
            entry["rating"].append(row["rating"])
            entry["score"].append(row["against strategic_ai"])
    return {
        round_number: {"rating": statistics.mean(values["rating"]), "score": statistics.mean(values["score"]), "comparison": comparison}
        for round_number, values in rated.items()
    }


def join(segments: list[Segment], named: dict[str, tuple[str, str]]) -> tuple[list[dict], list[dict]]:
    """The joined rounds, and a description of each segment."""

    rows: dict[int, dict] = {}
    described = []
    seconds_before = 0.0
    linked_end: float | None = None  # the previous segment's last rating, on the linked scale
    for segment in segments:
        settings = json.loads((RUNS_DIRECTORY / segment.run / "settings.json").read_text())
        logged = [row for row in read_metrics(segment.run) if row["round"] <= segment.rounds]
        ratings = comparison_ratings(segment.run, named)
        shift = 0.0
        if ratings and linked_end is not None and 0 in ratings:
            shift = linked_end - ratings[0]["rating"]

        for row in logged:
            overall = segment.first + row["round"]
            if row["round"] == 0 and overall in rows:
                continue  # the previous run's last round is the same network
            entry = rows.setdefault(overall, {"round": overall})
            entry["seconds"] = seconds_before + row["seconds"]
            # The logged weight where the run had a schedule, else the constant from its settings.
            entry["entropy bonus/weight"] = row.get("entropy bonus/weight", settings.get("entropy_weight"))
            for name, value in row.items():
                if name in ("round", "seconds"):
                    continue
                entry[RENAMED.get(name, name.replace("vs starting model", "vs the start of its run"))] = value
        for round_number, rated in ratings.items():
            if round_number > segment.rounds:
                continue
            entry = rows.setdefault(segment.first + round_number, {"round": segment.first + round_number})
            entry["rating/linked across comparisons"] = rated["rating"] + shift
            entry["comparison score/vs strategic_ai"] = rated["score"]

        last_rated = max((r for r in ratings if r <= segment.rounds), default=None)
        start = ratings.get(0, {}).get("rating")
        described.append(
            {
                "run": segment.run,
                "step": segment.step,
                "rounds": [segment.first, segment.last],
                "hours": (logged[-1]["seconds"] if logged else 0.0) / 3600,
                "entropy bonus": settings.get("entropy_weight"),
                "starting step size": settings.get("learning_rate"),
                "material hint": [settings.get("material_hint"), settings.get("material_hint_rounds")],
                "rated by": ratings[last_rated]["comparison"] if last_rated is not None else None,
                "rating at start, linked": None if start is None else start + shift,
                "rating at end, linked": None if last_rated is None else ratings[last_rated]["rating"] + shift,
            }
        )
        if last_rated is not None:
            linked_end = ratings[last_rated]["rating"] + shift
        seconds_before += logged[-1]["seconds"] if logged else 0.0

    # Ratings first, so the monitor draws them first.
    ordered = []
    for overall in sorted(rows):
        entry = rows[overall]
        front = {key: entry[key] for key in ("round", "seconds") if key in entry}
        rating = {key: value for key, value in entry.items() if key.startswith(("rating", "comparison score"))}
        rest = {key: value for key, value in entry.items() if key not in front and key not in rating}
        if "seconds" not in front:
            front["seconds"] = ordered[-1]["seconds"] if ordered else 0.0
        ordered.append({**front, **rating, **rest})
    return ordered, described


def write(name: str, rows: list[dict], described: list[dict], other: list[dict]) -> Path:
    folder = RUNS_DIRECTORY / f"lineage_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (folder / "segments.json").write_text(json.dumps({"before self-play": other, "segments": described}, indent=1) + "\n")
    return folder


def describe(described: list[dict], other: list[dict]) -> str:
    lines = []
    for entry in other:
        details = ", ".join(f"{key}={value}" for key, value in entry.items() if key not in ("step", "time"))
        lines.append(f"before: {entry['step']} ({details})")
    lines.append(f"{'rounds':>11}  {'run':22}{'hours':>7}{'entropy':>9}{'rating (linked)':>18}")
    for entry in described:
        first, last = entry["rounds"]
        start, end = entry["rating at start, linked"], entry["rating at end, linked"]
        rating = "-" if end is None else f"{start:.0f} -> {end:.0f}" if start is not None else f"{end:.0f}"
        lines.append(
            f"{first:>4} - {last:<4}  {entry['run']:22}{format_duration(entry['hours'] * 3600):>7}"
            f"{entry['entropy bonus']:>9g}{rating:>18}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", help="The network whose training to show, e.g. ml/runs/shaping_entropy/round_0150.pt")
    parser.add_argument("--name", help="Folder name after lineage_ (default: the last run's name)")
    parser.add_argument(
        "--ratings", nargs="*", default=[], metavar="RUN=COMPARISON/GROUP",
        help="Which comparison rated a run, for comparisons made before they recorded it",
    )
    args = parser.parse_args()
    named = dict(KNOWN_RATINGS)
    for item in args.ratings:
        run, _, where = item.partition("=")
        comparison, _, group = where.partition("/")
        named[run] = (comparison, group)
    segments, other = segments_of(args.checkpoint)
    if not segments:
        raise SystemExit(f"{args.checkpoint} has no self-play or league runs in its history")
    rows, described = join(segments, named)
    folder = write(args.name or segments[-1].run, rows, described, other)
    print(describe(described, other))
    print(f"\nJoined {len(rows)} rounds into {folder}. To look at them:")
    print(f"    .venv/bin/python -m ml monitor --run {folder.name}")


if __name__ == "__main__":
    main()
