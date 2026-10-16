"""The folder a training run writes to, and everything that goes into it.

Every trainer reports through a ``RunLog``, so every run looks the same::

    ml/runs/<name>/
        settings.json      what the run was started with
        metrics.jsonl      one line of numbers per training round
        round_0001.pt ...  the model after each round
        latest.pt          the newest model, with what is needed to resume
        best.pt            the round that scored best

``ml/runs/latest.pt`` is also kept pointing at the newest model of the most
recent run. That is the file the game's "Neural AI" plays, so a match started
while training is running shows the current state.

Metric names look like ``group/name``. The monitor draws one chart per group.
"""

import json
import shutil
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from ml.models import PolicyValueModel
from ml.models.checkpoint import save_checkpoint


RUNS_DIRECTORY = Path(__file__).resolve().parent.parent / "runs"
METRICS_FILE = "metrics.jsonl"


class RunLog:
    def __init__(self, name: str, settings, overwrite: bool = False, resume: bool = False):
        self.name = name
        self.folder = RUNS_DIRECTORY / name
        if resume:
            # Carry on where the run stopped: keep every file, and keep counting time from there.
            if not (self.folder / METRICS_FILE).exists():
                raise FileNotFoundError(f"There is no run named '{name}' to resume in {self.folder}")
            logged = read_metrics(self.folder)
            self.started = time.time() - (logged[-1]["seconds"] if logged else 0.0)
            return
        if (self.folder / METRICS_FILE).exists():
            if not overwrite:
                raise FileExistsError(
                    f"A run named '{name}' already exists in {self.folder}. "
                    "Pick another name, or pass overwrite to replace it."
                )
            shutil.rmtree(self.folder)
        self.folder.mkdir(parents=True, exist_ok=True)

        if is_dataclass(settings):
            settings = asdict(settings)
        (self.folder / "settings.json").write_text(json.dumps(settings, indent=2, default=str) + "\n")
        (self.folder / METRICS_FILE).touch()
        self.started = time.time()

    def seconds(self) -> float:
        return time.time() - self.started

    def say(self, message: str) -> None:
        print(f"[{format_duration(self.seconds()):>7}] {message}", flush=True)

    def log(self, round_number: int, metrics: dict[str, float], summary: str) -> None:
        """Record one round: a line in the metrics file and a line on the terminal."""

        entry = {"round": round_number, "seconds": round(self.seconds(), 1), **metrics}
        with (self.folder / METRICS_FILE).open("a") as file:
            file.write(json.dumps(entry) + "\n")
        self.say(summary)

    def save(
        self,
        round_number: int,
        model: PolicyValueModel,
        history: list[dict],
        training_state: dict | None = None,
    ) -> Path:
        """Save this round's model. Returns the path of its own numbered file."""

        path = save_checkpoint(self.folder / f"round_{round_number:04d}.pt", model, history)
        # Only the newest file carries the training state; it is the only one a run resumes from.
        save_checkpoint(self.folder / "latest.pt", model, history, training_state)
        shutil.copyfile(path, self.folder.parent / "latest.pt")
        return path

    def mark_best(self, path: Path) -> None:
        """Remember this round's model as the best of the run so far."""

        shutil.copyfile(path, self.folder / "best.pt")


def read_metrics(run: str | Path) -> list[dict]:
    """Every round logged so far, oldest first. ``run`` is a run name or its folder."""

    folder = Path(run)
    if not folder.is_dir():
        folder = RUNS_DIRECTORY / str(run)
    rounds = []
    for line in (folder / METRICS_FILE).read_text().splitlines():
        try:
            rounds.append(json.loads(line))
        except json.JSONDecodeError:
            # A line still being written when we looked; it will be complete next time.
            pass
    return rounds


def min_send_gap(checkpoint: str | Path) -> int:
    """The send limit (``min_send_gap``) of the run a checkpoint belongs to; 0 for none or no run.

    Measurements play a run's networks under the limit the run trained under, round 0 included.
    """

    settings = Path(checkpoint).parent / "settings.json"
    if not settings.is_file():
        return 0
    return int(json.loads(settings.read_text()).get("min_send_gap", 0))


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
