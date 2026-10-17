"""Put the final numbers of several runs side by side.

One column per run: its network's size and settings, how long it took, and
the metrics of its last logged round. No games are played; everything comes
from the runs' folders.

    .venv/bin/python -m ml summary capacity_current capacity_wider
    .venv/bin/python -m ml summary imitation_v1 stage10_imitation_seed1 --metrics agreement "move loss"

``--metrics`` keeps only metrics whose names contain one of the given words.
"""

import argparse
from pathlib import Path

from ml.models.checkpoint import load_checkpoint
from ml.training.runlog import RUNS_DIRECTORY, format_duration, read_metrics


# Settings that describe the size of a network; the others follow the encoders.
SIZE_SETTINGS = ("width", "layers", "heads", "route_layers")


def summarise(runs: list[str], metric_filter: list[str] | None = None) -> str:
    columns = [_column(run) for run in runs]
    metric_names = []
    for column in columns:
        for name in column["metrics"]:
            if name not in metric_names and (not metric_filter or any(word in name for word in metric_filter)):
                metric_names.append(name)

    rows = [
        ("weights", [f"{column['weights']:,}" if column["weights"] else "-" for column in columns]),
        *(
            (setting, [str(column["settings"].get(setting, "-")) for column in columns])
            for setting in SIZE_SETTINGS
        ),
        ("rounds", [str(column["round"]) for column in columns]),
        ("time", [format_duration(column["seconds"]) for column in columns]),
        *((name, [_number(column["metrics"].get(name)) for column in columns]) for name in metric_names),
    ]
    label_width = max(len(label) for label, _ in rows)
    widths = [max(len(Path(run).name), *(len(row[1][index]) for row in rows)) for index, run in enumerate(runs)]
    lines = [" " * label_width + "".join(f"  {Path(run).name:>{width}}" for run, width in zip(runs, widths))]
    for label, values in rows:
        lines.append(f"{label:{label_width}}" + "".join(f"  {value:>{width}}" for value, width in zip(values, widths)))
    return "\n".join(lines)


def _column(run: str) -> dict:
    folder = Path(run) if Path(run).is_dir() else RUNS_DIRECTORY / run
    logged = read_metrics(folder)
    last = logged[-1] if logged else {"round": 0, "seconds": 0.0}
    weights, settings = 0, {}
    if (folder / "latest.pt").exists():
        model = load_checkpoint(folder / "latest.pt").model
        weights = sum(parameter.numel() for parameter in model.parameters())
        settings = vars(model.config)
    metrics = {name: value for name, value in last.items() if name not in ("round", "seconds")}
    return {"round": last["round"], "seconds": last["seconds"], "weights": weights, "settings": settings,
            "metrics": metrics}


def _number(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and abs(value) < 10:
        return f"{value:.3f}"
    return f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="Run names in ml/runs/, or run folders")
    parser.add_argument("--metrics", nargs="+", metavar="WORD", help="Only metrics whose names contain one of these")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(summarise(args.runs, args.metrics))


if __name__ == "__main__":
    main()
