"""Imitation trainer: teach the network to play like a scripted AI.

1. The teacher plays many games on generated maps, and its decisions are
   recorded: the position it saw and the move it made.
2. The network is shown those positions and adjusted, a little at a time, to
   give the teacher's move a higher probability. Its value output is
   adjusted towards how each game ended for the teacher.
3. After every pass over the recorded decisions, the network is checked on
   games it was not trained on, and plays real matches against the scripted
   AIs.

A network with free amounts learns the exact units the teacher sent; one
with amount options learns the option nearest to them.

This is ordinary supervised learning: there is a right answer for every
example. It cannot make the network better than its teacher, but it gets a
sensible player quickly, which self-play can then improve.

Started through ``ml/training/train.py``:

    .venv/bin/python -m ml train imitation --run my_first_run
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import mse_loss

from ml.device import default_device, release_cached_memory
from ml.evaluation.evaluate import DEFAULT_OPPONENTS, FIRST_EVALUATION_SEED, evaluate
from ml.map_generator import generate_map
from ml.models import DEFAULT_MODEL, ModelOutput, PolicyValueModel
from ml.models.checkpoint import history_entry, init_model
from ml.models.policy import choose, log_probability
from ml.play.rollout import Matchup, RolloutPool, Seat
from ml.training.dataset import DecisionSet
from ml.training.runlog import RunLog


OUTCOME_VALUE = {"win": 1.0, "draw": 0.0, "loss": -1.0}


@dataclass(frozen=True)
class ImitationSettings:
    # ---- Teacher games ---------------------------------------------------------
    teacher: str = "strategic_ai"
    # One opponent per game, taken in turn. Listing the teacher here makes it
    # play itself, and then both seats are recorded.
    opponents: tuple[str, ...] = ("strategic_ai", "strategic_ai", "aggressive_ai", "careful_ai")
    games: int = 600
    # A long game contributes at most this many decisions per recorded seat,
    # picked at random, so long games do not crowd out short ones.
    decisions_per_game: int = 300
    # This share of the games is never trained on and is used to check progress.
    validation_share: float = 0.1

    # ---- Learning ----------------------------------------------------------------
    epochs: int = 20  # passes over the recorded decisions
    batch_size: int = 512
    # The step size shrinks smoothly from the first value to the second.
    learning_rate: float = 1e-3
    final_learning_rate: float = 1e-4
    value_weight: float = 0.5  # how much the value error counts next to the move error
    max_gradient: float = 1.0  # larger updates are scaled down to this size

    # ---- Feedback ----------------------------------------------------------------
    # After each epoch the network plays both seats on this many evaluation
    # maps against every scripted AI. 0 switches this off.
    evaluation_maps: int = 8
    evaluate_every: int = 1
    # When training ends, the best model plays the full evaluation and the table is printed.
    final_evaluation: bool = True

    # ---- Other -------------------------------------------------------------------
    seed: int = 0
    workers: int = 6


def train_imitation(
    run_name: str,
    settings: ImitationSettings = ImitationSettings(),
    init: str | Path = "random",
    model_name: str = DEFAULT_MODEL,
    overwrite: bool = False,
    device: torch.device | str | None = None,
) -> Path:
    """Run the whole thing and return the path of the best checkpoint."""

    device = torch.device(device) if device else default_device()
    log = RunLog(run_name, settings, overwrite)
    rng = np.random.default_rng(settings.seed)
    torch.manual_seed(settings.seed)

    checkpoint = init_model(init, model_name, seed=settings.seed)
    model = checkpoint.model.to(device)
    log.say(f"Run '{run_name}': imitating {settings.teacher}, starting from {init}, learning on {device}")

    training, validation = collect_teacher_games(settings, rng, log, model.free_amounts)
    log.say(f"Recorded {len(training):,} decisions to learn from and {len(validation):,} to check against")

    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    steps = settings.epochs * math.ceil(len(training) / settings.batch_size)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, settings.final_learning_rate)

    best_score = -1.0
    # Epoch 0: the starting model, measured the same way, before it has learned anything here.
    start = log.save(0, model, checkpoint.history)
    log.mark_best(start)
    metrics = check(model, validation, settings.batch_size, device)
    summary = f"before training  same move as teacher {metrics['agreement/all decisions']:.0%}"
    if settings.evaluation_maps > 0:
        scores, best_score = match_scores(str(start), settings)
        metrics.update(scores)
        summary += "  |  " + describe_scores(scores)
    log.log(0, in_chart_order(metrics), summary)

    for epoch in range(1, settings.epochs + 1):
        move_loss, value_loss = learn_one_pass(
            model, optimizer, schedule, training, settings.batch_size,
            settings.value_weight, settings.max_gradient, rng, device,
        )
        metrics = {
            "move loss/training": move_loss,
            "value loss/training": value_loss,
            **check(model, validation, settings.batch_size, device),
        }
        release_cached_memory(device)
        summary = (
            f"epoch {epoch:2d}/{settings.epochs}  move loss {metrics['move loss/training']:.3f} "
            f"(check {metrics['move loss/validation']:.3f})  "
            f"same move as teacher {metrics['agreement/all decisions']:.0%} "
            f"(sends {metrics['agreement/when teacher sends']:.0%}, waits {metrics['agreement/when teacher waits']:.0%})"
        )

        history = checkpoint.history + [
            history_entry(
                "imitation",
                teacher=settings.teacher,
                games=settings.games,
                decisions=len(training),
                epochs=epoch,
                run=run_name,
            )
        ]
        state = {"optimizer": optimizer.state_dict(), "epoch": epoch}
        path = log.save(epoch, model, history, state)

        evaluated = settings.evaluation_maps > 0 and (
            epoch % settings.evaluate_every == 0 or epoch == settings.epochs
        )
        if evaluated:
            scores, overall = match_scores(str(path), settings)
            metrics.update(scores)
            summary += "  |  " + describe_scores(scores)
            if overall > best_score:
                best_score = overall
                log.mark_best(path)
        elif settings.evaluation_maps == 0:
            # Without matches to go by, the newest model counts as the best.
            log.mark_best(path)

        log.log(epoch, in_chart_order(metrics), summary)

    best = log.folder / "best.pt"
    log.say(f"Done. Best model: {best}")
    if settings.final_evaluation:
        log.say("Full evaluation of the best model:")
        print(evaluate(str(best), workers=settings.workers).table(), flush=True)
    return best


def collect_teacher_games(
    settings: ImitationSettings,
    rng: np.random.Generator,
    log: RunLog,
    free_amounts: bool = False,
) -> tuple[DecisionSet, DecisionSet]:
    """Let the teacher play, and sort its decisions into a training and a validation set.

    ``free_amounts``: for a network with free amounts, which learns the units themselves.
    """

    matchups = teacher_matchups(settings.teacher, settings.opponents, settings.games, rng)
    with RolloutPool(settings.workers) as pool:
        trajectories = play_teacher_games(pool, matchups, settings.workers * 8, log)
    return split_decisions(trajectories, settings.decisions_per_game, settings.validation_share, rng, free_amounts)


def teacher_matchups(teacher: str, opponents: tuple[str, ...], games: int, rng: np.random.Generator) -> list[Matchup]:
    """Games for the teacher, against each opponent in turn and from each seat equally often.

    An opponent that is the teacher itself is recorded too.
    """

    matchups = []
    for number in range(games):
        opponent = opponents[number % len(opponents)]
        recorded = Seat(teacher, record=True)
        other = Seat(opponent, record=opponent == teacher)
        # Training maps come from below the seeds that are reserved for evaluation.
        definition = generate_map(int(rng.integers(FIRST_EVALUATION_SEED))).definition
        matchups.append(Matchup(definition, recorded, other) if number % 2 == 0 else Matchup(definition, other, recorded))
    return matchups


def play_teacher_games(
    pool: RolloutPool,
    matchups: list[Matchup],
    chunk: int,
    log: RunLog,
    max_seconds: float = 180.0,
) -> list[list]:
    """Play the games in chunks and return the recorded trajectories, one list per game."""

    games = []
    for start in range(0, len(matchups), chunk):
        played = pool.play(matchups[start : start + chunk], max_seconds=max_seconds)
        games += [record.trajectories for record in played]
        log.say(f"Teacher games: {len(games)}/{len(matchups)} played")
    return games


def split_decisions(
    games: list[list],
    decisions_per_game: int,
    validation_share: float,
    rng: np.random.Generator,
    free_amounts: bool = False,
) -> tuple[DecisionSet, DecisionSet]:
    """Sort the games' decisions into a training and a validation set, whole games at a time."""

    training, validation = DecisionSet(free_amounts), DecisionSet(free_amounts)
    validation_every = max(2, round(1 / validation_share)) if validation_share > 0 else 0
    held_out_games = set(range(validation_every - 1, len(games), validation_every)) if validation_every else set()
    # Too few games for the share to pick one: still hold out the last, so there is something to check against.
    if validation_every and not held_out_games and len(games) >= 2:
        held_out_games = {len(games) - 1}
    for number, trajectories in enumerate(games):
        held_out = number in held_out_games
        for trajectory in trajectories:
            outcome = np.full(len(trajectory), OUTCOME_VALUE[trajectory.outcome], dtype=np.float32)
            (validation if held_out else training).add(trajectory, {"outcome": outcome}, decisions_per_game, rng)
    return training, validation


def learn_one_pass(
    model: PolicyValueModel,
    optimizer: torch.optim.Optimizer,
    schedule,
    decisions: DecisionSet,
    batch_size: int,
    value_weight: float,
    max_gradient: float,
    rng: np.random.Generator,
    device: torch.device | str,
) -> tuple[float, float]:
    """One pass over the decisions, nudging the network towards the teacher's moves. Returns both losses."""

    model.train()
    move_loss = value_loss = seen = 0.0
    for batch in decisions.batches(batch_size, rng, device):
        output = model(batch.inputs)
        move_error, value_error = losses(output, batch)
        optimizer.zero_grad()
        (move_error + value_weight * value_error).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient)
        optimizer.step()
        schedule.step()
        move_loss += move_error.item() * len(batch)
        value_loss += value_error.item() * len(batch)
        seen += len(batch)
    return move_loss / seen, value_loss / seen


def match_scores(candidate: str, settings: ImitationSettings) -> tuple[dict[str, float], float]:
    """Play the scripted AIs on the evaluation maps. Returns chart data and the overall score."""

    results = evaluate(
        candidate,
        generated_maps=settings.evaluation_maps,
        bundled_maps=False,
        workers=settings.workers,
    )
    scores = {}
    for opponent in DEFAULT_OPPONENTS:
        part = results.summary(opponent=opponent)
        scores[f"score/vs {opponent}"] = part.score
        scores[f"wins/vs {opponent}"] = part.wins / part.games
    scores["score/overall"] = results.summary().score
    return scores, scores["score/overall"]


def describe_scores(scores: dict[str, float]) -> str:
    return "score " + " ".join(
        f"{opponent.removesuffix('_ai')} {scores[f'score/vs {opponent}']:.0%}" for opponent in DEFAULT_OPPONENTS
    )


def in_chart_order(metrics: dict[str, float]) -> dict[str, float]:
    """The monitor draws charts in the order the groups first appear: results before losses."""

    order = ("score", "wins", "agreement", "move loss", "value loss")
    return dict(sorted(metrics.items(), key=lambda item: order.index(item[0].split("/")[0])))


def losses(output: ModelOutput, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """How unlikely the network found the teacher's moves, and how wrong its value was."""

    move_error = -log_probability(output, batch.route, batch.amount).mean()
    value_error = mse_loss(output.value, batch.targets["outcome"])
    return move_error, value_error


def check(
    model: PolicyValueModel,
    decisions: DecisionSet,
    batch_size: int,
    device: torch.device | str,
) -> dict[str, float]:
    """Measure the network on decisions it was not trained on."""

    model.eval()
    move_loss = value_loss = always_zero = 0.0
    same = same_on_sends = same_on_waits = sends = total = 0
    with torch.no_grad():
        for batch in decisions.batches(batch_size, device=device):
            output = model(batch.inputs)
            move_error, value_error = losses(output, batch)
            move_loss += move_error.item() * len(batch)
            value_loss += value_error.item() * len(batch)
            always_zero += (batch.targets["outcome"] ** 2).sum().item()

            # The network's single most likely move, compared with the teacher's.
            route, amount = choose(output, greedy=True)
            teacher_sends = batch.route != 0
            agrees = (route == batch.route) & (~teacher_sends | (amount == batch.amount))

            same += int(agrees.sum())
            same_on_sends += int((agrees & teacher_sends).sum())
            same_on_waits += int((agrees & ~teacher_sends).sum())
            sends += int(teacher_sends.sum())
            total += len(batch)

    return {
        "move loss/validation": move_loss / total,
        "value loss/validation": value_loss / total,
        # How wrong a value output that always says 0 would be on the same decisions.
        "value loss/always saying 0": always_zero / total,
        "agreement/all decisions": same / total,
        "agreement/when teacher sends": same_on_sends / max(1, sends),
        "agreement/when teacher waits": same_on_waits / max(1, total - sends),
    }
