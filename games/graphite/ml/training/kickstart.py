"""Kickstart: a new network learns to play like a trained one, before self-play takes over.

A larger network (or one with new outputs) would otherwise have to repeat all
the self-play an existing network went through. Instead:

1. The teacher (a trained network) plays games against itself; a sample of
   the positions is kept, with where each send ended up.
2. The student is shown those positions and learns the teacher's whole
   opinion of each: its probability for every move (the Kullback-Leibler
   divergence) and its value. A student with a
   capture output also learns, for every send, whether its target was the
   sender's right after the fleet landed.
3. After every pass, the same is measured on games it never saw.

Unlike copying a scripted AI's single move (which is all a script shows),
the teacher's full probabilities say how close the alternatives were, so
the student picks up the teacher's style rather than a caricature of it.

Amount options the teacher does not know (added after it was trained) would
get no probability at all; the student would never try them. So on every
route where such options are on offer, they share ``new_option_share`` of
the route's amount probability, and the teacher's own proportions among the
others are kept.

A student with free amounts (ml/models/amounts.py) learns, on every route,
the teacher's probability for each number of units: options that send the
same units (one eighth and two eighths of 3 units are both 1) count as one
amount, with their probabilities added up.

Started through ``ml/training/train.py``; ``--init`` is the student before
learning, e.g. a fresh larger network made with ``ml checkpoint new``:

    .venv/bin/python -m ml checkpoint new ml/runs/student.pt --set width=128 layers=8 capture_head=yes
    .venv/bin/python -m ml train kickstart --run large_kickstart --init ml/runs/student.pt \\
        --teacher ml/runs/ab_plain/round_0400.pt
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import log_softmax, mse_loss

from ml.device import default_device, release_cached_memory
from ml.evaluation.evaluate import FIRST_EVALUATION_SEED
from ml.judge.progress import LiveLine, bar, count, duration
from ml.map_generator import generate_map
from ml.models import DEFAULT_MODEL, ModelOutput, amounts
from ml.models.checkpoint import history_entry, init_model, load_checkpoint
from ml.models.interface import readable_route_list
from ml.play.rollout import Matchup, RolloutPool, Seat
from ml.training.dataset import DecisionSet
from ml.training.runlog import RunLog
from ml.training.selfplay import _captured, capture_loss


TEACHER = "teacher"


@dataclass(frozen=True)
class KickstartSettings:
    teacher: str = "ml/runs/ab_plain/round_0400.pt"
    # ---- Teacher games -------------------------------------------------------------
    games: int = 1200  # against itself, both seats kept
    # Each seat of each game gives at most this many positions, picked at random.
    decisions_per_game: int = 100
    held_out_share: float = 0.1  # of the games, never learned from
    # ---- Learning --------------------------------------------------------------------
    epochs: int = 20
    patience: int = 3  # stop once the held-out divergence has not improved for this many passes
    batch_size: int = 512
    learning_rate: float = 1e-3  # shrinks smoothly to the next value
    final_learning_rate: float = 1e-5
    max_gradient: float = 1.0
    # Of each route's amount probability, the share given to options the teacher does not know.
    new_option_share: float = 0.05
    value_weight: float = 1.0  # towards the teacher's value
    capture_weight: float = 0.1  # towards where each send ended up (students with a capture output)
    # ---- Other -------------------------------------------------------------------------
    seed: int = 0
    workers: int = 6


def train_kickstart(
    run_name: str,
    settings: KickstartSettings = KickstartSettings(),
    init: str | Path = "random",
    model_name: str = DEFAULT_MODEL,
    overwrite: bool = False,
    device: torch.device | str | None = None,
) -> Path:
    device = torch.device(device) if device else default_device()
    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    log = RunLog(run_name, settings, overwrite)

    teacher = load_checkpoint(settings.teacher).model.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    start = init_model(init, model_name, seed=settings.seed)
    student, history = start.model.to(device), start.history
    if teacher.free_amounts:
        raise ValueError("The teacher must choose amount options; teaching from free amounts is not built yet")
    weights = lambda model: sum(parameter.numel() for parameter in model.parameters())  # noqa: E731
    log.say(
        f"Run '{run_name}': kickstarting {init} ({count(weights(student))} weights) "
        f"from {settings.teacher} ({count(weights(teacher))} weights), learning on {device}"
    )

    training, held_out = _teacher_positions(teacher, settings, rng, log, student.free_amounts)
    log.say(f"Positions: {count(len(training))} to learn from, {count(len(held_out))} held out")

    optimizer = torch.optim.Adam(student.parameters(), lr=settings.learning_rate)
    batches = training.batch_count(settings.batch_size)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings.epochs * batches, settings.final_learning_rate)
    before = _evaluate(student, teacher, held_out, settings, device)
    log.log(0, _metrics(before), "before learning   |  " + _describe(before))
    best_divergence, best_epoch = float("inf"), 0
    live = LiveLine()

    for epoch in range(1, settings.epochs + 1):
        started = time.monotonic()
        student.train()
        totals, seen = np.zeros(3), 0
        for number, batch in enumerate(training.batches(settings.batch_size, rng, device, equal_size=True), 1):
            divergence, value_error, capture_error = _losses(student, teacher, batch, settings)
            loss = divergence + settings.value_weight * value_error
            if capture_error is not None:
                loss = loss + settings.capture_weight * capture_error
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), settings.max_gradient)
            optimizer.step()
            schedule.step()
            totals += [divergence.item(), value_error.item(), 0.0 if capture_error is None else capture_error.item()]
            seen += 1
            spent = time.monotonic() - started
            live.show(
                f"epoch {epoch:2d}/{settings.epochs} {bar(number, batches)} {count(number)}/{count(batches)} batches"
                f"   divergence {totals[0] / seen:.4f}  value {totals[1] / seen:.4f}"
                f"   ({duration(spent)}, about {duration(spent / number * (batches - number))} left)"
            )
        live.show(f"epoch {epoch:2d}/{settings.epochs} checking on the held-out positions ...")
        check = _evaluate(student, teacher, held_out, settings, device)
        live.done()
        release_cached_memory(device)

        path = log.save(epoch, student, history + [history_entry("kickstarted", teacher=settings.teacher, epochs=epoch, run=run_name)])
        improved = check["divergence"] < best_divergence
        if improved:
            best_divergence, best_epoch = check["divergence"], epoch
            log.mark_best(path)
        waited = epoch - best_epoch
        log.log(
            epoch,
            {**_metrics(check), "divergence/training": totals[0] / seen},
            f"epoch {epoch:2d}/{settings.epochs}  |  " + _describe(check) + f"  ({duration(time.monotonic() - started)})"
            + ("  best so far" if improved else f"  no better for {waited}"),
        )
        if settings.patience and waited >= settings.patience:
            log.say(f"Stopping early: no improvement on the held-out positions for {waited} epochs (best: epoch {best_epoch}).")
            break

    log.say(f"Done. Best model (epoch {best_epoch}): {log.folder / 'best.pt'}")
    return log.folder / "best.pt"


def _teacher_positions(
    teacher, settings: KickstartSettings, rng: np.random.Generator, log: RunLog, free_amounts: bool = False
) -> tuple[DecisionSet, DecisionSet]:
    """The teacher's games against itself, sampled, split by game into learning and held-out positions.

    ``free_amounts``: for a student with free amounts, whose batches say what each option sends.
    """

    seat = Seat(TEACHER, record=True)
    matchups = [
        # Training maps come from below the seeds that are reserved for evaluation.
        Matchup(generate_map(int(rng.integers(FIRST_EVALUATION_SEED))).definition, seat, seat)
        for _ in range(settings.games)
    ]
    held_out_every = max(2, round(1 / settings.held_out_share))
    training, held_out = (DecisionSet(free_amounts, option_units=free_amounts) for _ in range(2))
    played, started = [0], time.monotonic()
    live = LiveLine()

    def finished(*_) -> None:
        played[0] += 1
        spent = time.monotonic() - started
        live.show(
            f"teacher games {bar(played[0], len(matchups))} {count(played[0])}/{count(len(matchups))}"
            f"   ({duration(spent)}, about {duration(spent / played[0] * (len(matchups) - played[0]))} left)"
        )

    with RolloutPool(settings.workers) as pool:
        # In chunks, so the positions are sampled as the games come in and whole games never pile up.
        chunk = settings.workers * 16
        for first in range(0, len(matchups), chunk):
            records = pool.play(
                matchups[first : first + chunk], {TEACHER: teacher}, seed=int(rng.integers(2**31)), on_finished=finished
            )
            for number, record in enumerate(records, first):
                part = held_out if number % held_out_every == held_out_every - 1 else training
                for trajectory in record.trajectories:
                    part.add(trajectory, {"captured": _captured(trajectory)}, settings.decisions_per_game, rng)
    live.done()
    log.say(f"Teacher games: {count(played[0])} played in {duration(time.monotonic() - started)}")
    return training, held_out


def target_amounts(teacher_output: ModelOutput, allowed: torch.Tensor, known: int, share: float) -> torch.Tensor:
    """The amount probabilities the student learns, per route: the teacher's, with ``share`` of
    each route's probability moved to the options it does not know (the last ones), where on offer.
    [batch, bases, bases, amount options]"""

    probabilities = torch.softmax(teacher_output.amount_logits, dim=-1)
    new = allowed[..., known:].float()
    offered = new.sum(dim=-1, keepdim=True)
    if not new.numel() or share <= 0:
        return probabilities
    spread = torch.cat((torch.zeros_like(probabilities[..., :known]), new / offered.clamp(min=1)), dim=-1)
    mixed = (1 - share) * probabilities + share * spread
    return torch.where(offered > 0, mixed, probabilities)


def _losses(student, teacher, batch, settings: KickstartSettings):
    """Divergence from the teacher over every move, the value's distance from the teacher's, and the capture error."""

    with torch.no_grad():
        taught = teacher(batch.inputs)
    output = student(batch.inputs)
    known = teacher.config.amount_options
    teacher_route = log_softmax(taught.route_logits, dim=1)
    student_route = log_softmax(output.route_logits, dim=1)
    route_divergence = (teacher_route.exp() * (teacher_route - student_route)).sum(dim=1)

    if output.amount_curves is not None:
        amount_divergence = free_amount_divergence(taught, output, batch.option_units)
    else:
        target = target_amounts(taught, batch.inputs.allowed, known, settings.new_option_share)
        student_amount = log_softmax(output.amount_logits, dim=-1)
        # Only where the target gives probability; elsewhere 0 * log 0 counts as 0.
        amount_divergence = torch.where(target > 0, target * (torch.log(target.clamp(min=1e-12)) - student_amount), 0.0).sum(dim=-1)
    # Each route's amount counts as often as the teacher would take that route.
    weighted = (teacher_route.exp()[:, 1:] * amount_divergence.flatten(start_dim=1)).sum(dim=1)
    divergence = (route_divergence + weighted).mean()
    value_error = mse_loss(output.value, taught.value)
    return divergence, value_error, capture_loss(output, batch)


def free_amount_divergence(teacher_output: ModelOutput, student_output: ModelOutput, option_units: torch.Tensor) -> torch.Tensor:
    """Per route, how far a student's free amounts are from a teacher's amount options. [batch, bases, bases]

    The teacher's probability of sending u units is the sum over its options that send u.
    The divergence sum(p(u) * (log p(u) - log q(u))) runs over those u; routes that cannot be
    used get 0. ``option_units`` is what each option sends ([batch, bases, bases, amount options]).
    """

    # The list is padded for the Apple GPU; the padding counts for nothing.
    position, source, target, real = readable_route_list(student_output.most > 0)
    taught = torch.softmax(teacher_output.amount_logits[position, source, target], dim=-1)  # [routes, options]
    units = option_units[position, source, target]
    offered = units > 0
    same = (units[:, :, None] == units[:, None, :]) & offered[:, None, :]
    merged = (same.float() * taught[:, None, :]).sum(dim=-1)  # each option's units, all options together
    options = units.shape[-1]
    learned = amounts.log_probability(
        student_output.amount_curves[position, source, target][:, None].expand(-1, options, -1, -1),
        student_output.most[position, source, target][:, None].expand(-1, options),
        student_output.needed[position, source, target][:, None].expand(-1, options),
        units.clamp(min=1),
    )
    per_option = torch.where(offered, taught * (torch.log(merged.clamp(min=1e-12)) - learned), 0.0)
    divergence = student_output.most.new_zeros(student_output.most.shape)
    return divergence.index_put((position, source, target), per_option.sum(dim=-1) * real, accumulate=True)


def _evaluate(student, teacher, decisions: DecisionSet, settings: KickstartSettings, device) -> dict[str, float]:
    student.eval()
    totals = {"divergence": 0.0, "value": 0.0, "capture": 0.0, "same favourite": 0.0}
    seen = captured = 0
    with torch.no_grad():
        for batch in decisions.batches(settings.batch_size, device=device):
            divergence, value_error, capture_error = _losses(student, teacher, batch, settings)
            n = len(batch)
            totals["divergence"] += divergence.item() * n
            totals["value"] += value_error.item() * n
            if capture_error is not None:
                totals["capture"] += capture_error.item() * n
                captured += n
            # How often the student's most likely first step (wait, or which route) is the teacher's.
            same = student(batch.inputs).route_logits.argmax(dim=1) == teacher(batch.inputs).route_logits.argmax(dim=1)
            totals["same favourite"] += same.float().sum().item()
            seen += n
    result = {name: total / max(seen, 1) for name, total in totals.items()}
    result["capture"] = totals["capture"] / captured if captured else float("nan")
    return result


def _metrics(check: dict[str, float]) -> dict[str, float]:
    metrics = {
        "divergence/held out": check["divergence"],
        "agreement/same favourite": check["same favourite"],
        "value error/vs the teacher": check["value"],
    }
    if not np.isnan(check["capture"]):
        metrics["capture error/held out"] = check["capture"]
    return metrics


def _describe(check: dict[str, float]) -> str:
    text = (
        f"held out: divergence {check['divergence']:.4f}, same favourite {check['same favourite']:.1%}, "
        f"value error {check['value']:.4f}"
    )
    if not np.isnan(check["capture"]):
        text += f", capture error {check['capture']:.3f} (50:50 is 0.693)"
    return text
