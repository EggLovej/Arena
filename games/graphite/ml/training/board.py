"""Training curves in TensorBoard, for one run or several side by side.

Run from the game directory, while training is running or afterwards:

    .venv/bin/python -m ml tensorboard --run free_scratch --run free_old_entropy

Then open the address it prints (http://localhost:6006/). Without --run it
shows every run. Ctrl+C stops it; training is not affected.

The runs' metrics files are copied into TensorBoard's own format in
``ml/runs/tensorboard/`` (rebuilt each time the command starts, holding only
the runs asked for), and the copies are kept up to date while it runs, so a
run in progress grows in the browser. A metric named ``score/vs careful_ai``
is the chart "score/vs careful_ai" in the section "score", one line per run.

On the x-axis, STEP is the round. RELATIVE is the hours the run has trained
(paused time not counted); WALL is the same counted from 1 January 1970, so
not useful. Smoothing, log scale (the button under each chart) and which runs
are shown (the regex box) are in the browser.
"""

import argparse
import math
import shutil
import time
from pathlib import Path

from ml.training.runlog import METRICS_FILE, RUNS_DIRECTORY, read_metrics


TENSORBOARD_DIRECTORY = RUNS_DIRECTORY / "tensorboard"
CHECK_SECONDS = 5.0
# Not metrics: every line has them, and they become the x-axes.
AXES = ("round", "seconds")


class Export:
    """One run's metrics as a TensorBoard event file in ``target``, kept up to date by ``update``."""

    def __init__(self, source: Path, target: Path):
        self.source = source
        self.target = target
        self.writer = None
        self.written = 0  # rounds of the metrics file already in the event file

    def update(self) -> int:
        """Write the rounds logged since the last call. Returns how many."""

        rounds = read_metrics(self.source)
        if len(rounds) < self.written:
            # The run was started again from scratch (--overwrite): start its copy again too.
            self.close()
            shutil.rmtree(self.target, ignore_errors=True)
            self.written = 0
        new = rounds[self.written :]
        if not new:
            return 0
        if self.writer is None:
            # Imported here: torch takes a moment, and the tests of other modules do not need it.
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(self.target)
        for entry in new:
            for name, value in entry.items():
                if name in AXES or not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                self.writer.add_scalar(name, value, global_step=entry["round"], walltime=entry["seconds"])
        self.writer.flush()
        self.written = len(rounds)
        return len(new)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def start_tensorboard(folder: Path, port: int) -> str:
    """Start TensorBoard on ``folder`` in the background of this process. Returns its address."""

    from tensorboard import program

    board = program.TensorBoard()
    # Every round's value is kept (by default TensorBoard keeps a random 1000 per line; 0, which
    # its help calls "keep all", keeps none with its fast loader), and the files are looked at
    # as often as they are written.
    board.configure(
        logdir=str(folder), port=port, samples_per_plugin={"scalars": 100_000}, reload_interval=CHECK_SECONDS
    )
    return board.launch()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", help="Name of a run in ml/runs/ (repeat for several; default every run)")
    parser.add_argument("--port", type=int, default=6006, help="Port of the page on this machine (default 6006)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    found = sorted(path.parent.name for path in RUNS_DIRECTORY.glob(f"*/{METRICS_FILE}"))
    runs = args.run or found
    unknown = [run for run in runs if run not in found]
    if unknown:
        raise SystemExit(f"No run named {', '.join(map(repr, unknown))}. Runs found: {', '.join(found) or 'none'}")

    shutil.rmtree(TENSORBOARD_DIRECTORY, ignore_errors=True)
    exports = [Export(RUNS_DIRECTORY / run, TENSORBOARD_DIRECTORY / run) for run in runs]
    for export in exports:
        export.update()
    address = start_tensorboard(TENSORBOARD_DIRECTORY, args.port)
    print(f"TensorBoard: {address}  ({len(runs)} run{'s' if len(runs) > 1 else ''}; Ctrl+C stops it)", flush=True)
    try:
        while True:
            time.sleep(CHECK_SECONDS)
            for export in exports:
                export.update()
    except KeyboardInterrupt:
        pass
    finally:
        for export in exports:
            export.close()


if __name__ == "__main__":
    main()
