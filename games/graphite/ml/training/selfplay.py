"""Self-play trainer: the network improves by playing against itself.

Every round has three steps:

1. **Play.** The current network plays a batch of games, picking moves by
   chance according to its own probabilities.
2. **Judge.** Each move is scored by how things went afterwards compared with
   what was expected beforehand. That score is called the advantage:
   positive means the move turned out better than expected.
3. **Learn.** Moves with a positive advantage are made a little more likely
   and the others a little less. The value output is corrected towards how
   each game really ended. The method that keeps these corrections small and
   safe is called PPO.

Unlike imitation there is no right answer to copy, only results. That is why
this can grow past any teacher.

Started through ``ml/training/train.py``:

    .venv/bin/python -m ml train selfplay --run from_scratch
    .venv/bin/python -m ml train selfplay --run from_teacher --init ml/runs/imitation_v1/best.pt
"""

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss

from game_types import BLUE, RED

from ml.device import default_device, release_cached_memory
from ml.encoding.state_encoder import BASE_FEATURES, GLOBAL_FEATURES, PAIR_FEATURES, UNITS_REFERENCE
from ml.evaluation.evaluate import (
    DEFAULT_OPPONENTS,
    FIRST_EVALUATION_SEED,
    Evaluation,
    MatchPlan,
    MatchResult,
    evaluate,
    evaluation_maps,
)
from ml.map_generator import generate_map
from ml.models import DEFAULT_MODEL, PolicyValueModel
from ml.models.checkpoint import history_entry, init_model, load_checkpoint
from ml.models.policy import capture_logit, entropy_parts, log_probability
from ml.play.rollout import WAIT, GameRecord, Matchup, RolloutPool, Seat, Trajectory
from ml.training.dataset import DecisionSet
from ml.training.imitation import OUTCOME_VALUE
from ml.training.runlog import RunLog


SELF = "self"  # in ``opponents``: the network plays both seats
START = "starting model"  # the frozen copy of the network this run began with

_UNITS_SHARE = GLOBAL_FEATURES.index("my_units_share")
_PRODUCTION_SHARE = GLOBAL_FEATURES.index("my_production_share")
_IS_MINE, _IS_ENEMY, _UNITS, _INCOMING_MINE = (
    BASE_FEATURES.index(name) for name in ("is_mine", "is_enemy", "units", "incoming_mine")
)
_PATH_TIME = PAIR_FEATURES.index("shortest_path_time")
_LOG_UNITS = math.log1p(UNITS_REFERENCE)  # the encoder's scale for unit counts
# The monitor draws charts in the order the groups first appear: results first.
_CHART_ORDER = (
    "score", "elo", "promotion match", "training wins", "decided games",
    "game length", "behaviour", "sends", "entropy", "entropy bonus", "learning problems", "policy change", "value error", "capture error",
)  # fmt: skip


@dataclass(frozen=True)
class TrainingSettings:
    """Settings shared by every trainer that learns from its own games."""

    # ---- Games -------------------------------------------------------------------
    rounds: int = 100
    games_per_round: int = 96
    # Each seat of each game contributes at most this many decisions to the
    # update, picked at random. Neighbouring ticks are nearly identical.
    decisions_per_game: int = 200
    decision_interval: int = 1  # the network decides every Nth tick
    # The networks of the run (the one learning, and the starting model it is checked against)
    # may send at most once every this many ticks, 10 being once a second, and wait in between;
    # the scripted AIs play as always. 0: no limit. Added 2026-10-05 for an ablation, see
    # history.md, "Acting once a second".
    min_send_gap: int = 0
    max_seconds: float = 180.0

    # ---- Judging moves -----------------------------------------------------------
    # A result one tick further away counts this much less. 0.999 halves the
    # weight of a result every 69 seconds of game time.
    discount: float = 0.999
    # How far ahead a move is judged. Near 1, by the rest of the game; near 0,
    # only by how the position looked one decision later.
    lookahead: float = 0.99
    # Until the value output has learned to judge positions, a rule of thumb
    # helps it: a larger share of production and units is better. Its weight
    # starts here and fades to zero over the given number of rounds.
    material_hint: float = 0.5
    material_hint_rounds: int = 50
    # Two more rules of thumb that stay on (added 2026-10-03; 0 turns them off). Both are
    # "potentials": a move is credited with how much it changed them, so giving something
    # back costs exactly what gaining it earned, and nothing can be collected twice. In
    # theory this never changes which strategy is best, only how soon a move gets credit.
    # production_potential: my share of the map's production (2 x share - 1, from -1 to +1),
    # so capturing, holding, and losing productive bases count at once.
    production_potential: float = 0.0
    # front_potential: how close my units are to the enemy. Each unit, at one of my bases or
    # on its way to any base, is as far away as the shortest path from there to the nearest
    # enemy base, as a share of the map's longest shortest path; the potential is minus the
    # average (from -1 to 0). Sending units forward earns credit, pulling them back costs.
    front_potential: float = 0.0

    # ---- Learning ----------------------------------------------------------------
    passes: int = 4  # how many times each round's decisions are gone over
    batch_size: int = 1024
    # The step size adjusts itself between rounds so that the policy moves by
    # about the target amount per round: a policy that is already confident
    # needs far smaller steps than an undecided one. It starts at the first
    # value and stays between the other two.
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 1e-3
    target_policy_change: float = 0.015
    # A round's learning stops as soon as the policy has moved this far from the one that played.
    max_policy_change: float = 0.03
    # A move's probability may change by this share per round before it stops counting.
    clip: float = 0.2
    # In the first rounds only the value output learns, so moves are not
    # judged by a value output that knows nothing yet. The value output shares
    # most of the network with the moves, so during those rounds the moves are
    # also held where they are, with this much force.
    value_only_rounds: int = 2
    hold_weight: float = 1.0
    value_weight: float = 0.5
    entropy_weight: float = 0.01  # reward for staying undecided, so it keeps trying things
    # The reward for staying undecided can shrink as training goes on (added 2026-10-04): it
    # halves every this many rounds of the run, from entropy_weight towards entropy_floor, and
    # never goes below the floor, so the network keeps trying things. 0 keeps it constant.
    # Measured: at 0.01 for 900 rounds the amounts became nearly random and progress stopped;
    # at 0.001 the network improved again.
    entropy_half_life: int = 0
    entropy_floor: float = 0.0
    # How much a network with a capture output (``capture_head``) learns from where each of
    # its sends ended up: was the target its own right after the fleet landed?
    capture_weight: float = 0.1
    max_gradient: float = 0.5

    # ---- Feedback ----------------------------------------------------------------
    # Every few rounds the network plays both seats on this many evaluation
    # maps against each scripted AI and against the model the run started with.
    evaluation_maps: int = 8
    evaluate_every: int = 5
    final_evaluation: bool = True

    # ---- Other -------------------------------------------------------------------
    seed: int = 0
    workers: int = 6
    # Where the network decides during games: "workers" (each worker process runs its own
    # copy on its CPU core) or "server" (one copy on the learning device answers every
    # worker's positions in batches; see ml/play/inference.py).
    inference: str = "workers"


@dataclass(frozen=True)
class SelfPlaySettings(TrainingSettings):
    # One opponent per game, taken in turn. "self" puts the network in both
    # seats; any other name is an AI file the network plays against.
    opponents: tuple[str, ...] = (SELF,)


@dataclass
class RoundReport:
    """What the opponent side has to say about a round."""

    metrics: dict[str, float] = field(default_factory=dict)
    summary: str = ""
    # True or False when the opponent side decides whether this round's model
    # becomes the run's best. None leaves that to the evaluation score.
    promoted: bool | None = None


class SelfPlayOpponents:
    """Who the network meets in training: itself, and any AI files named in the settings.

    The training loop asks this object for each round's games and tells it
    how the round went. The league provides a different one.
    """

    def __init__(self, settings: SelfPlaySettings):
        self.settings = settings

    def begin(self, starting_model, evaluation, pool: RolloutPool, log: RunLog, rng) -> None:
        """Called once, after the starting model has been evaluated."""

    def models(self) -> dict[str, PolicyValueModel]:
        """Other models that take part in the games, by player name."""

        return {}

    def matchups(self, rng: np.random.Generator) -> list[Matchup]:
        return make_matchups(self.settings, rng)

    def statistics(self, records: list[GameRecord]) -> dict[str, float]:
        return {**game_statistics(records), **behaviour(records)}

    def end_of_round(self, round_number, records, model, evaluation, pool: RolloutPool, rng) -> RoundReport:
        return RoundReport()

    def closing_report(self) -> str:
        return ""

    def state(self) -> dict:
        """What this side must remember to carry on after a restart. Saved with every round's checkpoint."""

        return {}

    def restore(self, state: dict, log: RunLog) -> None:
        """Called instead of ``begin`` when a run is resumed, with what ``state`` returned last time."""


def train_selfplay(
    run_name: str,
    settings: SelfPlaySettings = SelfPlaySettings(),
    init: str | Path = "random",
    model_name: str = DEFAULT_MODEL,
    overwrite: bool = False,
    device: torch.device | str | None = None,
    resume: bool = False,
    stop_after: int | None = None,
) -> Path:
    """Run the whole thing and return the path of the best checkpoint."""

    return run_training(
        run_name, settings, SelfPlayOpponents(settings), "self-play", init, model_name, overwrite, device,
        resume, stop_after,
    )


def run_training(
    run_name: str,
    settings: TrainingSettings,
    opponents: SelfPlayOpponents,
    label: str,
    init: str | Path = "random",
    model_name: str = DEFAULT_MODEL,
    overwrite: bool = False,
    device: torch.device | str | None = None,
    resume: bool = False,
    stop_after: int | None = None,
) -> Path:
    """The training loop shared by self-play and the league.

    ``resume`` carries on an existing run from its ``latest.pt``: weights,
    optimizer, step size, round number, and whatever the opponents remember
    (for the league: its champions, results, and ratings). ``stop_after`` ends this process
    after that many rounds, so a long run can be split over fresh processes
    (see ``ml/training/train.py --stop-after``).
    """

    device = torch.device(device) if device else default_device()
    log = RunLog(run_name, settings, overwrite, resume=resume)
    torch.manual_seed(settings.seed)

    def history(rounds: int) -> list[dict]:
        """The model's history after this many rounds of this run. The run's entry carries
        its send limit from round 0 on, so every checkpoint plays under it (``players.min_send_gap``)."""

        gap = {"min_send_gap": settings.min_send_gap} if settings.min_send_gap else {}
        entry = history_entry(label, rounds=rounds, games=rounds * settings.games_per_round, run=run_name, **gap)
        return checkpoint.history + [entry]

    if resume:
        checkpoint = load_checkpoint(log.folder / "latest.pt")
        state = checkpoint.training_state or {}
        first_round = state.get("round", 0) + 1
        # The entry for this run is written again, with the new round count, every round.
        checkpoint.history = [entry for entry in checkpoint.history if entry.get("run") != run_name]
        model = checkpoint.model.to(device)
        starting_model = load_checkpoint(log.folder / "round_0000.pt").model
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        learning_rate = state.get("learning_rate", settings.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        best_score = state.get("best_score", -1.0)
        opponents.restore(state.get("opponents", {}), log)
        # A different random stream from the first session's, so no maps are repeated.
        rng = np.random.default_rng([settings.seed, first_round])
        log.say(f"Run '{run_name}': resuming {label} at round {first_round}, learning on {device}")
    else:
        first_round = 1
        rng = np.random.default_rng(settings.seed)
        checkpoint = init_model(init, model_name, seed=settings.seed)
        model = checkpoint.model.to(device)
        starting_model = copy.deepcopy(model).cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
        learning_rate = settings.learning_rate
        best_score = -1.0
        log.say(f"Run '{run_name}': {label} starting from {init}, learning on {device}")
        best = log.save(0, model, history(0))
        log.mark_best(best)

    if settings.inference not in ("workers", "server"):
        raise ValueError(f"inference must be 'workers' or 'server', not '{settings.inference}'")
    stopped_early = False
    with RolloutPool(settings.workers, inference_device=device if settings.inference == "server" else None) as pool:
        if not resume:
            evaluation = None
            if settings.evaluation_maps > 0:
                # Round 0: how the starting model does, before it has learned anything here.
                evaluation = play_evaluation(pool, model, starting_model, settings)
                scores, best_score = evaluation_scores(evaluation)
                log.log(0, scores, "before training  |  " + describe_scores(scores))
            opponents.begin(starting_model, evaluation, pool, log, rng)

        for round_number in range(first_round, settings.rounds + 1):
            # 1. Play.
            records = pool.play(
                opponents.matchups(rng),
                {SELF: model, **opponents.models()},
                seed=int(rng.integers(2**31)),
                decision_interval=settings.decision_interval,
                max_seconds=settings.max_seconds,
            )

            # 2. Judge.
            hint = hint_weight(settings, round_number)
            decisions, value_errors = judged_decisions(records, settings, hint, rng, model.free_amounts)

            # 3. Learn.
            bonus = entropy_bonus(settings, round_number)
            learned = update(
                model,
                optimizer,
                decisions,
                settings,
                rng,
                device,
                learn_moves=round_number > settings.value_only_rounds,
                entropy_weight=bonus,
            )
            release_cached_memory(device)

            metrics = {
                **opponents.statistics(records), **learned, **value_errors,
                "learning rate": learning_rate, "entropy bonus/weight": bonus,
            }
            if round_number > settings.value_only_rounds:
                learning_rate = adapted_learning_rate(learning_rate, learned["policy change/per round"], settings)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
            summary = (
                f"round {round_number:3d}/{settings.rounds}  "
                f"{len(records)} games ({describe_games(records)})  "
                f"policy change {learned['policy change/per round']:.3f} at step size {metrics['learning rate']:.1e}  "
                f"undecided {learned['entropy/of the policy']:.2f}  "
                f"value error {value_errors['value error/of the network']:.3f} "
                f"(always saying 0: {value_errors['value error/always saying 0']:.3f})"
            )

            evaluation = overall = None
            if settings.evaluation_maps > 0 and (
                round_number % settings.evaluate_every == 0 or round_number == settings.rounds
            ):
                evaluation = play_evaluation(pool, model, starting_model, settings)
                scores, overall = evaluation_scores(evaluation)
                metrics.update(scores)
                summary += "  |  " + describe_scores(scores)

            report = opponents.end_of_round(round_number, records, model, evaluation, pool, rng)
            metrics.update(report.metrics)
            summary += report.summary
            if report.promoted is None:
                # Without a league, the best model is the one with the best evaluation score.
                is_best = (overall is not None and overall > best_score) or settings.evaluation_maps == 0
                if overall is not None and overall > best_score:
                    best_score = overall
            else:
                is_best = report.promoted

            # Everything needed to carry on from here, should this process stop.
            state = {
                "optimizer": optimizer.state_dict(),
                "round": round_number,
                "learning_rate": learning_rate,
                "best_score": best_score,
                "opponents": opponents.state(),
            }
            path = log.save(round_number, model, history(round_number), state)
            if is_best:
                log.mark_best(path)
            log.log(round_number, in_chart_order(metrics), summary)

            if stop_after and round_number - first_round + 1 >= stop_after and round_number < settings.rounds:
                stopped_early = True
                break

        closing = opponents.closing_report()
        if closing:
            print(closing, flush=True)

    best = log.folder / "best.pt"
    if stopped_early:
        log.say(f"Stopping this process after round {round_number}; continue with --resume")
        return best
    log.say(f"Done. Best model: {best}")
    if settings.final_evaluation:
        log.say("Full evaluation of the best model:")
        print(evaluate(str(best), workers=settings.workers).table(), flush=True)
    return best


def make_matchups(settings: SelfPlaySettings, rng: np.random.Generator) -> list[Matchup]:
    matchups = []
    for number in range(settings.games_per_round):
        opponent = settings.opponents[number % len(settings.opponents)]
        # Training maps come from below the seeds that are reserved for evaluation.
        definition = generate_map(int(rng.integers(FIRST_EVALUATION_SEED))).definition
        network = Seat(SELF, record=True, min_send_gap=settings.min_send_gap)
        if opponent == SELF:
            matchups.append(Matchup(definition, network, network))
        elif (number // len(settings.opponents)) % 2 == 0:
            matchups.append(Matchup(definition, network, Seat(opponent)))
        else:
            matchups.append(Matchup(definition, Seat(opponent), network))
    return matchups


def front_potential(trajectory: Trajectory) -> np.ndarray:
    """Minus how far my units are from the enemy, on average, at each decision: from -1 to 0.

    A unit at one of my bases, or on its way to any base, counts as far away as the
    shortest path from there to the nearest enemy base (0 for an enemy base itself), as a
    share of the map's longest shortest path. 0 where the enemy has no base or I have no units.
    """

    paths = trajectory.pair_features[..., _PATH_TIME].astype(np.float64)  # [bases, bases]
    features = trajectory.base_features.astype(np.float64)  # [decisions, bases, features]
    mine = features[..., _IS_MINE] > 0.5
    enemy = features[..., _IS_ENEMY] > 0.5
    units = np.expm1(features[..., _UNITS] * _LOG_UNITS) * mine + np.expm1(features[..., _INCOMING_MINE] * _LOG_UNITS)
    # Distance from every base to the nearest enemy base, per decision.
    distance = np.where(enemy[:, None, :], paths[None], np.inf).min(axis=2)
    known = np.isfinite(distance)
    total = (units * known).sum(axis=1)
    far = (units * np.where(known, distance, 0.0)).sum(axis=1)
    longest = max(paths.max(), 1e-9)
    potential = np.zeros(len(trajectory))
    has_both = enemy.any(axis=1) & (total > 0)
    potential[has_both] = -far[has_both] / (total[has_both] * longest)
    return potential


def behaviour(records: list[GameRecord]) -> dict[str, float]:
    """How the network's sends and units are spread, over the recorded seats of these games."""

    sends = own = decisions = 0
    away = []
    for record in records:
        for trajectory in record.trajectories:
            sent = np.flatnonzero(trajectory.source != WAIT)
            sends += len(sent)
            decisions += len(trajectory)
            own += int((trajectory.base_features[sent, trajectory.target[sent], _IS_MINE] > 0.5).sum())
            if len(trajectory):
                away.append(-front_potential(trajectory).mean())
    if not sends:
        return {}
    return {
        "behaviour/sends to own bases": own / sends,
        "sends/per game and seat": sends / max(sum(len(r.trajectories) for r in records), 1),
        "behaviour/units away from the front": float(np.mean(away)),
        # Only positions with a send on offer are recorded, so this is how often it held back.
        "behaviour/waits when it may send": 1 - sends / decisions,
    }


def entropy_bonus(settings: TrainingSettings, round_number: int) -> float:
    """This round's reward for staying undecided: halving every ``entropy_half_life`` rounds
    from ``entropy_weight`` towards ``entropy_floor`` (constant without a half-life)."""

    if settings.entropy_half_life <= 0:
        return settings.entropy_weight
    floor = min(settings.entropy_floor, settings.entropy_weight)
    halvings = (round_number - 1) / settings.entropy_half_life
    return floor + (settings.entropy_weight - floor) * 0.5**halvings


def hint_weight(settings: TrainingSettings, round_number: int) -> float:
    """The rule of thumb's weight in this round: full at round 1, fading to zero."""

    if settings.material_hint_rounds <= 0:
        return 0.0
    return settings.material_hint * max(0.0, 1 - (round_number - 1) / settings.material_hint_rounds)


def adapted_learning_rate(current: float, policy_change: float, settings: TrainingSettings) -> float:
    """Smaller steps after a round that moved the policy too far, larger after one that barely moved it."""

    if policy_change > settings.target_policy_change * 1.5:
        current /= 1.5
    elif policy_change < settings.target_policy_change / 1.5:
        current *= 1.5
    return min(max(current, settings.min_learning_rate), settings.max_learning_rate)


def judge_moves(
    trajectory: Trajectory,
    final_tick: int,
    settings: TrainingSettings,
    hint: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every decision of one seat in one game.

    Returns the advantage of each decision, and the target for the value
    output: the game's result (+1, 0, or -1), discounted by how many ticks
    away the end was.
    """

    result = OUTCOME_VALUE[trajectory.outcome]
    ticks = trajectory.tick.astype(np.float64)
    value_target = settings.discount ** (final_tick - ticks) * result

    # What was expected at each decision: the network's own value, plus the
    # rule of thumb while it still carries weight. Both shares run from 0 to
    # 1; an even position has 0.5 of each and counts as zero.
    units_share = trajectory.global_features[:, _UNITS_SHARE].astype(np.float64)
    production_share = trajectory.global_features[:, _PRODUCTION_SHARE].astype(np.float64)
    expected = trajectory.value.astype(np.float64) + hint * (units_share + production_share - 1.0)
    # The rules of thumb that stay on, added the same way.
    if settings.production_potential:
        expected = expected + settings.production_potential * (2.0 * production_share - 1.0)
    if settings.front_potential:
        expected = expected + settings.front_potential * front_potential(trajectory)

    # How much better things looked at the next decision than at this one.
    # For the last decision, "next" is the actual result.
    gap = np.diff(ticks)
    surprise = np.empty(len(ticks))
    surprise[:-1] = settings.discount**gap * expected[1:] - expected[:-1]
    surprise[-1] = value_target[-1] - expected[-1]

    # A move is credited with the surprises that follow it, the nearer ones more.
    advantage = np.empty(len(ticks))
    advantage[-1] = surprise[-1]
    carry = (settings.discount * settings.lookahead) ** gap
    for index in range(len(ticks) - 2, -1, -1):
        advantage[index] = surprise[index] + carry[index] * advantage[index + 1]
    return advantage.astype(np.float32), value_target.astype(np.float32)


def judged_decisions(
    records: list[GameRecord],
    settings: TrainingSettings,
    hint: float,
    rng: np.random.Generator,
    free_amounts: bool = False,
) -> tuple[DecisionSet, dict[str, float]]:
    """Judge every recorded game and collect a sample of its decisions for learning.

    Also returns how wrong the value output was during these games, next to
    how wrong a value output that always says 0 would have been. The network
    is only judging positions well if it beats that. ``free_amounts``: the
    network learning has free amounts.
    """

    judged = []
    for record in records:
        for trajectory in record.trajectories:
            if len(trajectory):
                judged.append((trajectory, *judge_moves(trajectory, record.ticks, settings, hint)))

    # Advantages are rescaled to a common size, so the step size means the
    # same whether the games were close or one-sided.
    every = np.concatenate([advantage for _, advantage, _ in judged])
    mean, spread = every.mean(), every.std() + 1e-8

    decisions = DecisionSet(free_amounts)
    for trajectory, advantage, value_target in judged:
        decisions.add(
            trajectory,
            {
                "advantage": (advantage - mean) / spread,
                "value_target": value_target,
                "log_probability": trajectory.log_probability,
                "captured": _captured(trajectory),
            },
            settings.decisions_per_game,
            rng,
        )

    predicted = np.concatenate([trajectory.value for trajectory, _, _ in judged])
    targets = np.concatenate([value_target for _, _, value_target in judged])
    value_errors = {
        "value error/of the network": float(np.mean((predicted - targets) ** 2)),
        "value error/always saying 0": float(np.mean(targets**2)),
    }
    return decisions, value_errors


def _captured(trajectory: Trajectory) -> np.ndarray:
    """Whether each send's target was ours once it landed (1 or 0); NaN where unknown."""

    if trajectory.captured is None:
        return np.full(len(trajectory), np.nan, dtype=np.float32)
    return trajectory.captured


def capture_loss(output, batch) -> torch.Tensor | None:
    """Cross-entropy of the capture output against where the sends ended up; None if there is nothing to learn."""

    if output.capture_logits is None or "captured" not in batch.targets:
        return None
    labels = batch.targets["captured"]
    known = ~torch.isnan(labels) & (batch.route > 0)
    if not known.any():
        return None
    logits = capture_logit(output, batch.route, batch.amount)
    return binary_cross_entropy_with_logits(logits[known], labels[known])


def update(
    model: PolicyValueModel,
    optimizer: torch.optim.Optimizer,
    decisions: DecisionSet,
    settings: TrainingSettings,
    rng: np.random.Generator,
    device: torch.device | str,
    learn_moves: bool = True,
    entropy_weight: float | None = None,
) -> dict[str, float]:
    """One round of learning (PPO). Returns numbers describing what changed.

    ``entropy_weight`` is this round's reward for staying undecided (see ``entropy_bonus``);
    by default the settings' starting value.
    """

    if entropy_weight is None:
        entropy_weight = settings.entropy_weight
    # A round's learning can, rarely, run into non-finite numbers (seen once, round 190 of
    # entropy_decay, on the Apple GPU). A batch whose loss or gradients are not finite is
    # skipped; if the weights end up broken anyway, the whole round's learning is undone.
    before = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
    skipped = 0
    model.train()
    moved_far_enough = False
    capture_total = capture_seen = 0.0
    for _ in range(settings.passes):
        change = clipped = route_undecided = amount_undecided = seen = 0.0
        for batch in decisions.batches(settings.batch_size, rng, device, equal_size=True):
            output = model(batch.inputs)
            new_log_probability = log_probability(output, batch.route, batch.amount)
            # How much more (above 0) or less (below 0) likely each move is now
            # than when it was played, as a logarithm, and as a plain ratio.
            shift = new_log_probability - batch.targets["log_probability"]
            ratio = torch.exp(shift)
            advantage = batch.targets["advantage"]

            value_loss = mse_loss(output.value, batch.targets["value_target"])
            route_spread, amount_spread = (part.mean() for part in entropy_parts(output))
            spread = route_spread + amount_spread
            if learn_moves:
                # The heart of PPO: gains from pushing a move's probability beyond
                # the clip range are ignored, so no single round can overdo it.
                limited = torch.clamp(ratio, 1 - settings.clip, 1 + settings.clip)
                move_loss = -torch.min(ratio * advantage, limited * advantage).mean()
                loss = move_loss + settings.value_weight * value_loss - entropy_weight * spread
            else:
                # Only the value output should learn. Any shift in the moves is
                # penalised, because correcting the value would otherwise drag
                # the moves along through the parts of the network they share.
                loss = settings.value_weight * value_loss + settings.hold_weight * (shift**2).mean()
            captures = capture_loss(output, batch)
            if captures is not None:
                loss = loss + settings.capture_weight * captures
                capture_total += captures.item()
                capture_seen += 1
            optimizer.zero_grad()
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_gradient)):
                optimizer.zero_grad()
                skipped += 1
                continue
            optimizer.step()

            count = len(batch)
            with torch.no_grad():
                change += -shift.mean().item() * count
                clipped += ((ratio - 1).abs() > settings.clip).float().mean().item() * count
            route_undecided += route_spread.item() * count
            amount_undecided += amount_spread.item() * count
            seen += count

            # Checked after every batch, not only at the end of a pass: a
            # confident policy can move a long way within a single pass.
            if learn_moves and seen >= 2 * settings.batch_size and change / seen > settings.max_policy_change:
                moved_far_enough = True
                break
        if moved_far_enough:
            break

    undone = not all(torch.isfinite(parameter).all() for parameter in model.parameters())
    if undone:
        model.load_state_dict(before[0])
        optimizer.load_state_dict(before[1])
    seen = max(seen, 1)
    learned = {
        "entropy/of the policy": (route_undecided + amount_undecided) / seen,
        # Its two parts: which route (or waiting), and which amount for the route.
        "entropy/of the route choice": route_undecided / seen,
        "entropy/of the amounts": amount_undecided / seen,
        "policy change/per round": change / seen,
        "clipped share": clipped / seen,
        "decisions learned from": float(len(decisions)),
    }
    if skipped or undone:
        learned["learning problems/batches skipped"] = float(skipped)
        learned["learning problems/round undone"] = float(undone)
    if capture_seen:
        # Cross-entropy; always saying "50:50" scores 0.69.
        learned["capture error/of the network"] = capture_total / capture_seen
    return learned


def play_evaluation(
    pool: RolloutPool,
    model: PolicyValueModel,
    starting_model: PolicyValueModel,
    settings: TrainingSettings,
) -> Evaluation:
    """Both seats on the evaluation maps against each scripted AI and the starting model."""

    plans = [
        MatchPlan(map_set, map_key, seat, opponent, 0)
        for opponent in (*DEFAULT_OPPONENTS, START)
        for map_set, map_key in evaluation_maps(settings.evaluation_maps, bundled_maps=False)
        for seat in (RED, BLUE)
    ]
    matchups = []
    for plan in plans:
        definition = generate_map(int(plan.map_key)).definition
        gap = settings.min_send_gap
        opponent = Seat(plan.opponent, min_send_gap=gap if plan.opponent == START else 0)
        seats = {plan.candidate_seat: Seat(SELF, min_send_gap=gap), -plan.candidate_seat: opponent}
        matchups.append(Matchup(definition, seats[RED], seats[BLUE]))

    # The same seed every time, so rounds are compared on equal luck.
    records = pool.play(matchups, {SELF: model, START: starting_model}, seed=settings.seed)
    results = []
    for plan, record in zip(plans, records):
        outcome = "draw" if record.winner is None else "win" if record.winner == plan.candidate_seat else "loss"
        results.append(MatchResult(plan, outcome, record.ticks * 0.1))
    return Evaluation("current network", results)


def evaluation_scores(results: Evaluation) -> tuple[dict[str, float], float]:
    """Score against each opponent as chart data, and the overall score."""

    scores = {
        f"score/vs {opponent}": results.summary(opponent=opponent).score
        for opponent in (*DEFAULT_OPPONENTS, START)
    }
    return scores, results.summary().score


def describe_scores(scores: dict[str, float]) -> str:
    return "score " + " ".join(
        f"{name.removeprefix('score/vs ').removesuffix('_ai')} {score:.0%}" for name, score in scores.items()
    )


def in_chart_order(metrics: dict[str, float]) -> dict[str, float]:
    def position(name: str) -> int:
        group = name.split("/")[0]
        return _CHART_ORDER.index(group) if group in _CHART_ORDER else len(_CHART_ORDER)

    return dict(sorted(metrics.items(), key=lambda item: position(item[0])))


def game_statistics(records: list[GameRecord]) -> dict[str, float]:
    metrics = {}
    mirror = [record for record in records if record.matchup.red.player == record.matchup.blue.player]
    if mirror:
        metrics["decided games/in self-play"] = sum(record.winner is not None for record in mirror) / len(mirror)
        metrics["game length/self-play, seconds"] = sum(record.ticks for record in mirror) / len(mirror) * 0.1

    against: dict[str, list[bool]] = {}
    for record in records:
        if record.matchup.red.player == record.matchup.blue.player:
            continue
        network_is_red = record.matchup.red.player == SELF
        opponent = record.matchup.blue.player if network_is_red else record.matchup.red.player
        against.setdefault(opponent, []).append(record.winner == (RED if network_is_red else BLUE))
    for opponent, wins in against.items():
        metrics[f"training wins/vs {opponent}"] = sum(wins) / len(wins)
    return metrics


def describe_games(records: list[GameRecord]) -> str:
    decided = sum(record.winner is not None for record in records) / len(records)
    seconds = sum(record.ticks for record in records) / len(records) * 0.1
    return f"{decided:.0%} decided, {seconds:.0f}s average"
