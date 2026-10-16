"""Step 3 of the judge loop: train the judge on recorded positions.

Ordinary supervised learning. Each position's label is how the game ended
for the player whose position it is: win, draw, or loss. The judge learns
the three probabilities with cross-entropy. After every epoch it is checked
on the held-out games; the epoch that did best there is kept as ``best.pt``.
At the end, the test games give the verdict:

- **test loss**, next to the baseline: always guessing the overall mix of
  wins, draws, and losses, without looking at the position (a judge that
  does not beat it has learned nothing);
- **score error**, the squared difference between the expected result
  (win 1, draw ½, loss 0) and the actual one, for the judge and for the
  playing network's value output on the same positions: is the new judge
  more precise than the old one?
- **calibration**: of the positions the judge gave 70–80%, how many were won?

    .venv/bin/python -m ml judge-train --data round1 --run judge1
    .venv/bin/python -m ml judge-train --data round1 --run judge1 --start-from ml/runs/deep_from_imitation/round_0500.pt
    .venv/bin/python -m ml judge-train --data round1 --run judge1 --set layers=6 route_features=0
    .venv/bin/python -m ml judge-train --data round1 round2 --run judge2      several data runs together

Two guards against learning the training games by heart (see ml/plan.md,
"The judge"): weight decay, a gentle pull of every weight towards zero, and
early stopping: training ends once the held-out loss has not improved for
``patience`` epochs, and the best epoch is the one kept.

``--start-from`` copies a playing network's understanding of positions (the
attention layers) into the new judge, so it does not start from nothing.
The run's folder (``ml/runs/RUN/``) can be watched with ``python -m ml monitor --run RUN``.
"""

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import cross_entropy

from ml.device import default_device, release_cached_memory
from ml.judge.games import DATA_DIRECTORY, load_positions
from ml.judge.progress import LiveLine, bar, count, duration
from ml.models import build_model
from ml.models.checkpoint import history_entry, load_checkpoint, model_settings, save_checkpoint
from ml.models.judge import OUTCOMES, GraphJudge
from ml.training.dataset import DecisionSet
from ml.training.runlog import RunLog


OUTCOME_INDEX = {outcome: index for index, outcome in enumerate(OUTCOMES)}
SCORE_OF = torch.tensor([1.0, 0.5, 0.0])  # win, draw, loss as a result


@dataclass(frozen=True)
class JudgeSettings:
    epochs: int = 20
    batch_size: int = 512
    # The step size shrinks smoothly from the first value to the second over the run.
    learning_rate: float = 1e-3
    final_learning_rate: float = 1e-5
    max_gradient: float = 1.0
    # Each step shrinks every weight (not the biases and normalisations) by this share of the step size.
    weight_decay: float = 0.01
    # Stop once the held-out loss has not improved for this many epochs; 0 never stops early.
    patience: int = 4
    seed: int = 0


def train_judge(
    run_name: str,
    data: str | list[str],
    settings: JudgeSettings = JudgeSettings(),
    init: str = "random",
    architecture: dict | None = None,
    start_from: str | None = None,
    compare_with: str | None = None,
    overwrite: bool = False,
    device: str | None = None,
) -> Path:
    """Train a judge and return the path of its best epoch."""

    device = torch.device(device) if device else default_device()
    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    data = [data] if isinstance(data, str) else list(data)
    log = RunLog(run_name, {**asdict(settings), "data": data, "init": init, "start_from": start_from}, overwrite)

    print("\n=== Training judge ===", flush=True)
    live = LiveLine()
    started = time.monotonic()
    sets = {}
    for split in ("training", "held_out", "test"):
        live.show(f"loading the {split.replace('_', '-')} positions of {', '.join(data)} from {DATA_DIRECTORY} ...")
        sets[split] = _decision_set(data, split, rng)
    live.done(
        f"positions: training {count(len(sets['training']))}, held out {count(len(sets['held_out']))}, "
        f"test {count(len(sets['test']))}   (loaded in {duration(time.monotonic() - started)})"
    )

    if init == "random":
        judge = build_model("graph_judge", **(architecture or {}))
        history = [history_entry("random start", model="graph_judge", seed=settings.seed)]
    else:
        start = load_checkpoint(init)
        judge, history = start.model, start.history
    if not isinstance(judge, GraphJudge):
        raise ValueError(f"{init} is not a judge")
    if start_from:
        copied = copy_matching_weights(load_checkpoint(start_from).model, judge)
        print(f"started from {start_from}: copied {copied} of {len(judge.state_dict())} weight tensors", flush=True)
    judge.to(device)

    baseline, mix = _baseline_loss(sets["training"], sets["held_out"])
    print(
        f"baseline loss {baseline:.3f}: always guessing the overall mix ({mix[0]:.0%} wins, {mix[1]:.0%} draws, "
        f"{mix[2]:.0%} losses) without looking at the position. The judge has to get well below this.",
        flush=True,
    )
    batches = sets["training"].batch_count(settings.batch_size)
    stopping = f", stopping early after {settings.patience} epochs without improvement" if settings.patience else ""
    print(
        f"At most {settings.epochs} epochs of {count(batches)} batches each{stopping}; after every epoch the loss on "
        f"the held-out games, which the judge never trains on, is what counts.\n",
        flush=True,
    )

    # Weight decay on the weights only: shrinking biases and normalisations helps nothing.
    decayed = [parameter for parameter in judge.parameters() if parameter.dim() >= 2]
    kept = [parameter for parameter in judge.parameters() if parameter.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decayed, "weight_decay": settings.weight_decay}, {"params": kept, "weight_decay": 0.0}],
        lr=settings.learning_rate,
    )
    steps = settings.epochs * math.ceil(len(sets["training"]) / settings.batch_size)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, settings.final_learning_rate)
    best_loss, best_path, best_epoch = float("inf"), None, 0

    for epoch in range(1, settings.epochs + 1):
        started = time.monotonic()
        judge.train()
        total, seen = 0.0, 0
        for number, batch in enumerate(sets["training"].batches(settings.batch_size, rng, device, equal_size=True), 1):
            loss = cross_entropy(judge.outcome_logits(batch.inputs), batch.targets["outcome"].long())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(judge.parameters(), settings.max_gradient)
            optimizer.step()
            schedule.step()
            total += loss.item() * len(batch)
            seen += len(batch)
            spent = time.monotonic() - started
            live.show(
                f"epoch {epoch:2d}/{settings.epochs} training  {bar(number, batches)} {count(number)}/{count(batches)} batches"
                f"   train loss so far {total / seen:.3f}   ({duration(spent)}, about {duration(spent / number * (batches - number))} left)"
            )
        training_loss = total / seen
        live.show(f"epoch {epoch:2d}/{settings.epochs} checking on the held-out games ...")
        held_out_loss = _loss(judge, sets["held_out"], settings.batch_size, device)
        release_cached_memory(device)

        path = save_checkpoint(
            log.folder / f"round_{epoch:04d}.pt", judge, history + [history_entry("judge training", epochs=epoch, data=data, run=run_name)]
        )
        if held_out_loss < best_loss:
            best_loss, best_path, best_epoch = held_out_loss, path, epoch
            shutil.copyfile(path, log.folder / "best.pt")
        waited = epoch - best_epoch
        shutil.copyfile(path, log.folder / "latest.pt")
        live.done()  # clear the live line; the run log prints the epoch's lasting line
        log.log(
            epoch,
            {"loss/training": training_loss, "loss/held out": held_out_loss, "loss/baseline": baseline},
            f"epoch {epoch:2d}/{settings.epochs}   train loss {training_loss:.3f}   held-out loss {held_out_loss:.3f}"
            f"   ({duration(time.monotonic() - started)} per epoch)"
            + ("   best so far" if not waited else f"   no better for {waited} epoch{'s' if waited > 1 else ''}"),
        )
        if settings.patience and waited >= settings.patience:
            print(
                f"\nStopping early: the held-out loss has not improved for {waited} epochs "
                f"(best: epoch {best_epoch}, {best_loss:.3f}).",
                flush=True,
            )
            break

    print("\n=== Testing the best epoch ===", flush=True)
    live.show("judging the test games (and the playing network's value output on them) ...")
    best = load_checkpoint(log.folder / "best.pt").model.to(device)
    reference = load_checkpoint(compare_with).model.to(device) if compare_with else None
    report = test_report(best, sets["test"], settings.batch_size, device, reference, baseline)
    live.done(f"test games: {count(report['positions'])} positions")
    print(report["text"], flush=True)
    (log.folder / "test.json").write_text(json.dumps({key: value for key, value in report.items() if key != "text"}, indent=1))
    print(f"\nBest epoch: {best_path.name} -> {log.folder / 'best.pt'}", flush=True)
    return log.folder / "best.pt"


def copy_matching_weights(source: torch.nn.Module, target: torch.nn.Module) -> int:
    """Copy every weight tensor with the same name and shape; returns how many."""

    own = target.state_dict()
    matching = {name: value for name, value in source.state_dict().items() if name in own and own[name].shape == value.shape}
    target.load_state_dict({**own, **matching})
    return len(matching)


def test_report(judge, decisions: DecisionSet, batch_size: int, device, reference=None, baseline: float | None = None) -> dict:
    """Test loss, score error (also for a reference network's value output), and calibration."""

    judge.eval()
    predicted, actual, reference_scores, losses = [], [], [], []
    with torch.no_grad():
        for batch in decisions.batches(batch_size, device=device):
            logits = judge.outcome_logits(batch.inputs)
            outcome = batch.targets["outcome"].long()
            losses.append(cross_entropy(logits, outcome, reduction="sum").item())
            probabilities = torch.softmax(logits, dim=1).cpu()
            predicted.append(probabilities[:, 0] + 0.5 * probabilities[:, 1])
            actual.append(SCORE_OF[outcome.cpu()])
            if reference is not None:
                reference_scores.append((reference(batch.inputs).value.cpu() + 1) / 2)
    predicted, actual = torch.cat(predicted), torch.cat(actual)
    report = {
        "test loss": sum(losses) / len(actual),
        "score error": float(((predicted - actual) ** 2).mean()),
        "positions": len(actual),
    }
    lines = [f"test loss {report['test loss']:.3f}" + (f"   (baseline, guessing the mix: {baseline:.3f})" if baseline else "")]
    lines.append(f"score error (squared, result 1 / ½ / 0): judge {report['score error']:.4f}")
    if reference is not None:
        report["score error, reference"] = float(((torch.cat(reference_scores) - actual) ** 2).mean())
        lines[-1] += f"   playing network's value output {report['score error, reference']:.4f}"
    lines.append("calibration (expected result -> actual result):")
    report["calibration"] = []
    for low in np.arange(0.0, 1.0, 0.1):
        inside = (predicted >= low) & (predicted < low + 0.1 if low < 0.85 else predicted <= 1.0)
        if inside.sum() == 0:
            continue
        said, got, number = float(predicted[inside].mean()), float(actual[inside].mean()), int(inside.sum())
        report["calibration"].append({"said": said, "got": got, "positions": number})
        lines.append(f"  said {low:.0%}-{low + 0.1:.0%}: average {said:5.1%} -> scored {got:5.1%}   ({count(number)} positions)")
    report["text"] = "\n".join(lines)
    return report


def _decision_set(data: list[str], split: str, rng: np.random.Generator) -> DecisionSet:
    decisions = DecisionSet()
    for trajectory in (trajectory for name in data for trajectory in load_positions(name, split)):
        outcome = np.full(len(trajectory), OUTCOME_INDEX[trajectory.outcome], dtype=np.float32)
        decisions.add(trajectory, {"outcome": outcome})
    if not len(decisions):
        raise ValueError(f"No {split} positions in {', '.join(data)} ({DATA_DIRECTORY})")
    return decisions


def _baseline_loss(training: DecisionSet, held_out: DecisionSet) -> tuple[float, np.ndarray]:
    """The held-out loss of always guessing the training mix of win, draw, and loss; and that mix."""

    mix = np.bincount(training.target("outcome").astype(int), minlength=3) / len(training)
    held = held_out.target("outcome").astype(int)
    return float(-np.log(np.clip(mix[held], 1e-9, None)).mean()), mix


def _loss(judge, decisions: DecisionSet, batch_size: int, device) -> float:
    judge.eval()
    total = 0.0
    with torch.no_grad():
        for batch in decisions.batches(batch_size, device=device):
            total += cross_entropy(judge.outcome_logits(batch.inputs), batch.targets["outcome"].long(), reduction="sum").item()
    return total / len(decisions)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, nargs="+", help="Folder names in ml/runs/judge_data/ (from ml judge-games); several are combined")
    parser.add_argument("--run", required=True, help="Folder name in ml/runs/ for the judge")
    parser.add_argument("--init", default="random", help='"random" (the default) or a judge checkpoint to continue')
    parser.add_argument("--set", nargs="+", default=[], metavar="SETTING=VALUE", help="Judge architecture, e.g. layers=6 route_features=0")
    parser.add_argument("--start-from", help="Copy the attention layers of this playing network into the new judge")
    parser.add_argument(
        "--compare-with",
        default="ml/runs/deep_from_imitation/round_0500.pt",
        help="A playing network whose value output is scored on the test positions next to the judge",
    )
    parser.add_argument("--overwrite", action="store_true")
    for setting in fields(JudgeSettings):
        default = setting.default
        parser.add_argument("--" + setting.name.replace("_", "-"), type=type(default), default=default, metavar="N", help=f"default: {default}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = JudgeSettings(**{setting.name: getattr(args, setting.name) for setting in fields(JudgeSettings)})
    train_judge(
        args.run,
        args.data,
        settings,
        init=args.init,
        architecture=model_settings("graph_judge", args.set),
        start_from=args.start_from,
        compare_with=args.compare_with or None,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
