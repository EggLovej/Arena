"""League trainer: self-play with a memory and a gate.

Plain self-play only ever meets the current network. Two things can go wrong
with that. It can forget how to beat what it used to beat, and it can go in
circles: A beats B, B beats C, C beats A, with no real progress.

The league adds three things to the same training loop:

- **Past versions as opponents.** Every promoted version, called a champion,
  stays in a pool. Part of each round's games are played against that pool,
  more often against the ones the network currently does worst against.
- **Scripted AIs as opponents.** A fixed reference that never changes.
- **A gate.** Every few rounds the network plays a match against the reigning
  champion. Only if it clearly wins does it become the new champion, and only
  champions are saved as the run's best model.

It also keeps a rating (Elo) for every champion and scripted AI, worked out
from all the matches between them, so progress is one number on one scale.

Started through ``ml/training/train.py``:

    .venv/bin/python -m ml train league --run my_league --init ml/runs/imitation_v1/best.pt

A league run can be split over several processes like self-play
(``--stop-after N``, then ``--resume``). Its champions are reloaded from their
``round_XXXX.pt`` files, so those must not be deleted while the run goes on.
"""

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from game_types import BLUE, RED

from ml.evaluation.evaluate import DEFAULT_OPPONENTS, FIRST_EVALUATION_SEED, Evaluation, evaluation_maps
from ml.map_generator import generate_map
from ml.models import DEFAULT_MODEL, PolicyValueModel
from ml.models.checkpoint import load_checkpoint
from ml.play.rollout import GameRecord, Matchup, RolloutPool, Seat
from ml.training.runlog import RunLog
from ml.training.selfplay import SELF, START, RoundReport, SelfPlayOpponents, TrainingSettings, run_training


LEARNER = "current network"
ANCHOR_RATING = 1000.0  # the scripted AIs average this by definition


@dataclass(frozen=True)
class LeagueSettings(TrainingSettings):
    # ---- League: who it plays ----------------------------------------------------
    # Shares of each round's games. What is left over goes to the scripted AIs.
    self_share: float = 0.5
    champion_share: float = 0.3
    scripted_opponents: tuple[str, ...] = DEFAULT_OPPONENTS
    # Only the newest champions stay in the pool of opponents.
    max_champions: int = 8

    # ---- League: promotion -------------------------------------------------------
    promotion_every: int = 5  # rounds between matches against the champion
    promotion_games: int = 128
    # Wins count 1 and draws 1/2. At 128 games, a network no better than the
    # champion reaches this score by luck about one time in eight.
    promotion_score: float = 0.55


@dataclass
class Champion:
    name: str
    model: PolicyValueModel
    round: int  # the training round it was promoted in; 0 for the starting model


class League(SelfPlayOpponents):
    def __init__(self, settings: LeagueSettings):
        self.settings = settings
        self.champions: list[Champion] = []
        # (first player, second player) -> [points scored by the first, games played]
        self.results: dict[tuple[str, str], list[float]] = {}
        self.ratings: dict[str, float] = {}
        # The network's recent score against each opponent it has trained against.
        self.recent: dict[str, float] = {}
        self.promotions: list[dict] = []
        self.folder: Path | None = None

    def begin(self, starting_model, evaluation, pool: RolloutPool, log: RunLog, rng) -> None:
        self.folder = log.folder
        self.champions = [Champion("champion 0", starting_model, 0)]

        # The scripted AIs play each other once. That fixes the scale the ratings are measured on.
        scripted = self.settings.scripted_opponents
        pairs = [(a, b) for index, a in enumerate(scripted) for b in scripted[index + 1 :]]
        maps = evaluation_maps(max(4, self.settings.evaluation_maps), bundled_maps=False)
        plans = [(a, b, key, a_is_red) for a, b in pairs for _, key in maps for a_is_red in (True, False)]
        matchups = [
            Matchup(generate_map(int(key)).definition, Seat(a if a_is_red else b), Seat(b if a_is_red else a))
            for a, b, key, a_is_red in plans
        ]
        for (a, b, _, a_is_red), record in zip(plans, pool.play(matchups, seed=self.settings.seed)):
            self._add_result(self.results, a, b, _points(record, RED if a_is_red else BLUE))

        if evaluation is not None:
            self._add_evaluation(self.results, "champion 0", evaluation)
        self.ratings = fit_ratings(self.results, scripted)
        self._save()
        log.say("League ratings at the start: " + self._rating_line())

    def models(self) -> dict[str, PolicyValueModel]:
        return {champion.name: champion.model for champion in self._pool()}

    def matchups(self, rng: np.random.Generator) -> list[Matchup]:
        settings = self.settings
        mirror_games = round(settings.games_per_round * settings.self_share)
        champion_games = round(settings.games_per_round * settings.champion_share)
        scripted_games = settings.games_per_round - mirror_games - champion_games
        if not settings.scripted_opponents:
            champion_games, scripted_games = champion_games + scripted_games, 0

        opponents = [SELF] * mirror_games
        opponents += self._pick([champion.name for champion in self._pool()], champion_games, rng)
        opponents += self._pick(list(settings.scripted_opponents), scripted_games, rng)

        network = Seat(SELF, record=True)
        matchups = []
        for number, opponent in enumerate(opponents):
            # Training maps come from below the seeds that are reserved for evaluation.
            definition = generate_map(int(rng.integers(FIRST_EVALUATION_SEED))).definition
            if opponent == SELF:
                matchups.append(Matchup(definition, network, network))
            elif number % 2 == 0:
                matchups.append(Matchup(definition, network, Seat(opponent)))
            else:
                matchups.append(Matchup(definition, Seat(opponent), network))
        return matchups

    def statistics(self, records: list[GameRecord]) -> dict[str, float]:
        metrics = {}
        mirror = [record for record in records if _opponent(record) == SELF]
        if mirror:
            metrics["decided games/in self-play"] = sum(record.winner is not None for record in mirror) / len(mirror)
            metrics["game length/self-play, seconds"] = sum(record.ticks for record in mirror) / len(mirror) * 0.1

        wins: dict[str, list[bool]] = {}
        for record in records:
            opponent = _opponent(record)
            if opponent != SELF:
                group = opponent if opponent in self.settings.scripted_opponents else "past champions"
                wins.setdefault(group, []).append(record.winner == _network_seat(record))
        for group in (*self.settings.scripted_opponents, "past champions"):
            if group in wins:
                metrics[f"training wins/vs {group}"] = sum(wins[group]) / len(wins[group])
        return metrics

    def end_of_round(self, round_number, records, model, evaluation, pool: RolloutPool, rng) -> RoundReport:
        self._remember(records)
        if round_number % self.settings.promotion_every:
            return RoundReport(promoted=False)

        champion = self.champions[-1]
        points, games = self._promotion_match(model, champion, pool, rng)
        score = points / games

        # Rate the current network as if it were a league member, from this
        # match and from its evaluation games this round.
        trial = {pair: list(result) for pair, result in self.results.items()}
        self._add_result(trial, LEARNER, champion.name, points, games)
        if evaluation is not None:
            self._add_evaluation(trial, LEARNER, evaluation)
        ratings = fit_ratings(trial, self.settings.scripted_opponents)
        metrics = {
            "promotion match/score against the champion": score,
            "promotion match/needed for promotion": self.settings.promotion_score,
            "elo/current network": ratings[LEARNER],
        }

        promoted = score >= self.settings.promotion_score
        if promoted:
            # The current network joins the league under its own name, results and all.
            name = f"champion {len(self.champions)}"
            self.champions.append(Champion(name, copy.deepcopy(model).cpu(), round_number))
            self.results = {
                tuple(name if player == LEARNER else player for player in pair): result
                for pair, result in trial.items()
            }
            self.ratings = fit_ratings(self.results, self.settings.scripted_opponents)
            summary = f"  |  {score:.0%} against {champion.name}: promoted to {name}"
        else:
            summary = (
                f"  |  {score:.0%} against {champion.name}, "
                f"{self.settings.promotion_score:.0%} needed: not promoted"
            )
        # Both ratings come from the same working-out, so they can be compared directly.
        metrics["elo/champion"] = ratings[LEARNER if promoted else champion.name]
        summary += f" (rating {ratings[LEARNER]:.0f}, champion {metrics['elo/champion']:.0f})"

        self.promotions.append(
            {"round": round_number, "against": champion.name, "score": score, "games": games, "promoted": promoted}
        )
        self._save()
        return RoundReport(metrics, summary, promoted)

    def closing_report(self) -> str:
        lines = ["League table:", f"  {'player':16}{'rating':>8}   promoted in round"]
        rounds = {champion.name: champion.round for champion in self.champions}
        for name, rating in sorted(self.ratings.items(), key=lambda item: -item[1]):
            lines.append(f"  {name:16}{rating:8.0f}   {rounds.get(name, '')}")
        return "\n".join(lines)

    def state(self) -> dict:
        # Champions are stored as names and rounds: their weights are already in the round files.
        return {
            "champions": [[champion.name, int(champion.round)] for champion in self.champions],
            "results": [
                [str(first), str(second), float(points), int(games)]
                for (first, second), (points, games) in self.results.items()
            ],
            "recent": {str(name): float(score) for name, score in self.recent.items()},
            "promotions": [dict(promotion, score=float(promotion["score"])) for promotion in self.promotions],
        }

    def restore(self, state: dict, log: RunLog) -> None:
        if not state.get("champions"):
            raise ValueError(
                f"Run '{log.name}' was saved before league runs could be resumed, so its league is lost. "
                "Start a new run, for example from its latest.pt."
            )
        self.folder = log.folder
        self.champions = []
        for name, round_number in state["champions"]:
            path = log.folder / f"round_{round_number:04d}.pt"
            if not path.exists():
                raise FileNotFoundError(f"{name} is missing: {path} was deleted, and the league needs it to carry on")
            self.champions.append(Champion(name, load_checkpoint(path).model, round_number))
        self.results = {(first, second): [points, games] for first, second, points, games in state["results"]}
        self.recent = dict(state["recent"])
        self.promotions = list(state["promotions"])
        self.ratings = fit_ratings(self.results, self.settings.scripted_opponents)
        log.say(f"League restored with {len(self.champions)} champions. Ratings: " + self._rating_line())

    def _pool(self) -> list[Champion]:
        return self.champions[-self.settings.max_champions :]

    def _pick(self, names: list[str], count: int, rng: np.random.Generator) -> list[str]:
        """Choose opponents, favouring the ones the network has recently done worst against."""

        if not names or count <= 0:
            return []
        weights = np.array([(1 - self.recent.get(name, 0.5)) ** 2 + 0.05 for name in names])
        # Plain strings, not NumPy's: player names end up in saved checkpoints, which only take plain data.
        return [str(name) for name in rng.choice(names, size=count, p=weights / weights.sum())]

    def _remember(self, records: list[GameRecord]) -> None:
        points: dict[str, list[float]] = {}
        for record in records:
            opponent = _opponent(record)
            if opponent != SELF:
                points.setdefault(opponent, []).append(_points(record, _network_seat(record)))
        for opponent, scored in points.items():
            # A running average that leans on the newest rounds.
            before = self.recent.get(opponent, 0.5)
            self.recent[opponent] = 0.7 * before + 0.3 * sum(scored) / len(scored)

    def _promotion_match(self, model, champion: Champion, pool: RolloutPool, rng) -> tuple[float, int]:
        """The current network against the champion on fresh maps, from both seats."""

        matchups = []
        for _ in range(self.settings.promotion_games // 2):
            definition = generate_map(int(rng.integers(FIRST_EVALUATION_SEED))).definition
            matchups.append(Matchup(definition, Seat(SELF), Seat(champion.name)))
            matchups.append(Matchup(definition, Seat(champion.name), Seat(SELF)))
        records = pool.play(
            matchups,
            {SELF: model, champion.name: champion.model},
            seed=int(rng.integers(2**31)),
            max_seconds=self.settings.max_seconds,
        )
        return sum(_points(record, _network_seat(record)) for record in records), len(records)

    @staticmethod
    def _add_result(results: dict, first: str, second: str, points: float, games: int = 1) -> None:
        entry = results.setdefault((first, second), [0.0, 0])
        entry[0] += points
        entry[1] += games

    def _add_evaluation(self, results: dict, name: str, evaluation: Evaluation) -> None:
        """Add a player's evaluation games. The starting model there is champion 0 here."""

        for opponent in (*DEFAULT_OPPONENTS, START):
            summary = evaluation.summary(opponent=opponent)
            rival = "champion 0" if opponent == START else opponent
            if summary.games and rival != name:
                self._add_result(results, name, rival, summary.score * summary.games, summary.games)

    def _rating_line(self) -> str:
        ordered = sorted(self.ratings.items(), key=lambda item: -item[1])
        return ", ".join(f"{name} {rating:.0f}" for name, rating in ordered)

    def _save(self) -> None:
        """Write the league's state next to the run's checkpoints, as something a person can read."""

        state = {
            "champions": [
                {
                    "name": champion.name,
                    "round": champion.round,
                    "file": f"round_{champion.round:04d}.pt",
                    "rating": round(self.ratings.get(champion.name, ANCHOR_RATING)),
                }
                for champion in self.champions
            ],
            "ratings": {name: round(rating) for name, rating in sorted(self.ratings.items())},
            "promotion_matches": self.promotions,
            "results": [[first, second, points, games] for (first, second), (points, games) in self.results.items()],
        }
        (self.folder / "league.json").write_text(json.dumps(state, indent=2) + "\n")


def train_league(
    run_name: str,
    settings: LeagueSettings = LeagueSettings(),
    init: str | Path = "random",
    model_name: str = DEFAULT_MODEL,
    overwrite: bool = False,
    device: torch.device | str | None = None,
    resume: bool = False,
    stop_after: int | None = None,
) -> Path:
    """Run the whole thing and return the path of the best checkpoint: the last champion."""

    if settings.min_send_gap:
        raise ValueError("The league does not support min_send_gap yet; use self-play")
    return run_training(
        run_name, settings, League(settings), "league", init, model_name, overwrite, device, resume, stop_after
    )


def fit_ratings(
    results: dict[tuple[str, str], list[float]],
    anchors: tuple[str, ...] = (),
) -> dict[str, float]:
    """Find the ratings that best explain every recorded match at once.

    A rating difference predicts a score: equal ratings mean 50%, 100 points
    more mean about 64%, 200 about 76%, 400 about 91%. The ratings returned
    are the ones whose predictions match each player's actual total. The
    anchors are shifted to average 1000, which fixes the scale.
    """

    players = sorted({player for pair in results for player in pair})
    played: dict[str, list[tuple[str, float, float]]] = {player: [] for player in players}
    for (first, second), (points, games) in results.items():
        played[first].append((second, points, games))
        played[second].append((first, games - points, games))

    def expected(own: float, other: float) -> float:
        return 1 / (1 + 10 ** ((other - own) / 400))

    rating = dict.fromkeys(players, 0.0)
    scale = math.log(10) / 400
    for _ in range(300):
        for player in players:
            # One imaginary win and one loss against an average player keep a
            # rating finite even for someone who has won or lost every game.
            even = expected(rating[player], 0.0)
            scored, predicted, leverage = 1.0, 2 * even, 2 * even * (1 - even)
            for other, points, games in played[player]:
                chance = expected(rating[player], rating[other])
                scored += points
                predicted += games * chance
                leverage += games * chance * (1 - chance)
            step = (scored - predicted) / (scale * leverage)
            rating[player] += max(-100.0, min(100.0, step))

    reference = [rating[name] for name in anchors if name in rating] or list(rating.values())
    shift = ANCHOR_RATING - sum(reference) / len(reference) if reference else 0.0
    return {player: value + shift for player, value in rating.items()}


def _network_seat(record: GameRecord) -> int:
    return RED if record.matchup.red.player == SELF else BLUE


def _opponent(record: GameRecord) -> str:
    """Who the current network played in this game; "self" if it held both seats."""

    red, blue = record.matchup.red.player, record.matchup.blue.player
    if red == blue:
        return SELF
    return blue if red == SELF else red


def _points(record: GameRecord, seat: int) -> float:
    if record.winner is None:
        return 0.5
    return 1.0 if record.winner == seat else 0.0
