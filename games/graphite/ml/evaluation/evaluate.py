"""Measure a player against fixed opponents on fixed maps.

The maps are ones training never sees: the hand-made 1v1 maps in ``maps/``
plus generated maps from seeds reserved for evaluation. Every pairing is
played from both seats, so an unfair map cannot flatter either side.

A player is named by a short text (all of them are in ml/play/players.py):

    untrained            a network with random weights (untrained:7 picks the seed)
    path/to/model.pt     a saved checkpoint
    strategic_ai         any AI file in ais/

From the game directory:

    .venv/bin/python -m ml evaluate --candidate untrained
    .venv/bin/python -m ml evaluate --candidate ml/runs/demo/random.pt --opponents strategic_ai
    .venv/bin/python -m ml evaluate --candidate strategic_ai --quick

Every game the candidate loses to ``aggressive_ai`` (or to the opponents named
with ``--save-losses-against``) is saved to ``ml/runs/losses/``, to be played
again with ``python -m ml rematch`` or watched with ``python -m ml watch``.
"""

import argparse
import os
import statistics
import sys
import time
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

from game import Game
from game_types import BLUE, RED, MapDefinition
from loading import available_maps, load_map

from ml.map_generator import generate_map
from ml.play.players import UNTRAINED, Controller, load_player


# Generated maps from these seeds are for evaluation only. Training must draw
# its maps from seeds below the first one.
FIRST_EVALUATION_SEED = 900_000
DEFAULT_GENERATED_MAPS = 30
DEFAULT_OPPONENTS = ("aggressive_ai", "careful_ai", "strategic_ai")
# A strong player should never lose to these; such a loss is saved for a closer look.
DEFAULT_SAVE_LOSSES_AGAINST = ("aggressive_ai",)
MATCH_SECONDS = 180.0
TRAINING = "network in training"  # how ``evaluate_model`` names the network it measures


@dataclass(frozen=True)
class MatchPlan:
    """One game to play: which map, which seat the candidate takes, against whom."""

    map_set: str  # "bundled" or "generated"
    map_key: str  # map name, or the seed as text
    candidate_seat: int
    opponent: str
    repeat: int


@dataclass(frozen=True)
class MatchResult:
    plan: MatchPlan
    outcome: str  # "win", "draw", or "loss", from the candidate's side
    seconds: float  # game time when the match ended
    saved_to: str = ""  # the saved game's file, if it was saved (see ml/evaluation/saved_games.py)


@dataclass(frozen=True)
class Summary:
    games: int
    wins: int
    draws: int
    losses: int
    score: float  # wins count 1, draws count 1/2
    margin: float  # the score would likely land within this much if replayed (95%)
    decided: float  # share of games that ended before the time limit
    median_seconds: float


class Evaluation:
    def __init__(self, candidate: str, results: list[MatchResult]):
        self.candidate = candidate
        self.results = results

    def summary(self, opponent: str | None = None, map_set: str | None = None) -> Summary:
        """Totals over all games, or only those against one opponent or on one map set."""

        chosen = [
            result
            for result in self.results
            if opponent in (None, result.plan.opponent) and map_set in (None, result.plan.map_set)
        ]
        games = len(chosen)
        wins = sum(result.outcome == "win" for result in chosen)
        draws = sum(result.outcome == "draw" for result in chosen)
        points = wins + draws / 2
        # Two imaginary wins and two losses keep the margin honest when every game went the same way.
        adjusted = (points + 2) / (games + 4)
        return Summary(
            games=games,
            wins=wins,
            draws=draws,
            losses=games - wins - draws,
            score=points / games if games else 0.0,
            margin=1.96 * (adjusted * (1 - adjusted) / (games + 4)) ** 0.5,
            decided=1 - draws / games if games else 0.0,
            median_seconds=statistics.median(result.seconds for result in chosen) if chosen else 0.0,
        )

    def table(self) -> str:
        opponents = sorted({result.plan.opponent for result in self.results})
        map_sets = sorted({result.plan.map_set for result in self.results})
        rows = [(f"vs {opponent}", self.summary(opponent=opponent)) for opponent in opponents]
        rows += [(f"on {map_set} maps", self.summary(map_set=map_set)) for map_set in map_sets]
        rows.append(("overall", self.summary()))

        lines = [f"{'':22}{'games':>6}{'wins':>6}{'draws':>7}{'losses':>8}{'score':>16}{'decided':>9}{'median length':>15}"]
        for label, s in rows:
            score = f"{s.score:6.1%} ± {s.margin:4.1%}"
            lines.append(
                f"{label:22}{s.games:6d}{s.wins:6d}{s.draws:7d}{s.losses:8d}{score:>16}"
                f"{s.decided:9.0%}{s.median_seconds:14.0f}s"
            )
        return "\n".join(lines)


def evaluate(
    candidate: str,
    opponents: Iterable[str] = DEFAULT_OPPONENTS,
    generated_maps: int = DEFAULT_GENERATED_MAPS,
    bundled_maps: bool = True,
    repeats: int = 1,
    greedy: bool = False,
    workers: int | None = None,
    max_seconds: float = MATCH_SECONDS,
    on_progress: Callable[[list[MatchResult], int], None] | None = None,
    skip_generated: int = 0,
    save_losses_against: Iterable[str] = (),
    save_directory: str | Path | None = None,
) -> Evaluation:
    """Play every pairing and return the results.

    ``on_progress`` is called after each finished game with the results so far
    and the total number of games. Games the candidate loses against an
    opponent in ``save_losses_against`` are saved to ``save_directory``
    (default ``ml/runs/losses/``).
    """

    plans = [
        MatchPlan(map_set, map_key, seat, opponent, repeat)
        for opponent in opponents
        for map_set, map_key in evaluation_maps(generated_maps, bundled_maps, skip_generated)
        for seat in (RED, BLUE)
        for repeat in range(repeats)
    ]
    saving = (tuple(save_losses_against), str(save_directory) if save_directory else None)
    jobs = [(plan, candidate, greedy, max_seconds, saving) for plan in plans]
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 2)

    results: list[MatchResult] = []
    if workers == 1:
        finished = map(_play, jobs)
        for result in finished:
            results.append(result)
            if on_progress:
                on_progress(results, len(plans))
    else:
        with Pool(workers, initializer=_prepare_worker) as pool:
            for result in pool.imap_unordered(_play, jobs):
                results.append(result)
                if on_progress:
                    on_progress(results, len(plans))

    # Finishing order depends on timing; sort so the same run always reports the same list.
    results.sort(key=lambda result: repr(result.plan))
    return Evaluation(candidate, results)


def evaluate_model(
    pool,
    model,
    generated_maps: int,
    opponents: Iterable[str] = DEFAULT_OPPONENTS,
    models: dict | None = None,
    min_send_gap: int = 0,
    seed: int = 0,
) -> Evaluation:
    """``evaluate`` for a network in training, played by its trainer's ``RolloutPool``: both
    seats on the first ``generated_maps`` evaluation maps against each opponent.

    ``models`` holds the opponents that are networks, by name. Every network here plays under
    ``min_send_gap``; the scripted AIs play as always. The same seed every time compares
    rounds on equal luck.
    """

    # Imported here, as the rollouts import this module.
    from ml.play.rollout import Matchup, Seat

    networks = {TRAINING: model, **(models or {})}
    plans = [
        MatchPlan(map_set, map_key, seat, opponent, 0)
        for opponent in opponents
        for map_set, map_key in evaluation_maps(generated_maps, bundled_maps=False)
        for seat in (RED, BLUE)
    ]
    matchups = []
    for plan in plans:
        seats = {
            plan.candidate_seat: Seat(TRAINING, min_send_gap=min_send_gap),
            -plan.candidate_seat: Seat(plan.opponent, min_send_gap=min_send_gap if plan.opponent in networks else 0),
        }
        matchups.append(Matchup(generate_map(int(plan.map_key)).definition, seats[RED], seats[BLUE]))

    results = []
    for plan, record in zip(plans, pool.play(matchups, networks, seed=seed)):
        outcome = "draw" if record.winner is None else "win" if record.winner == plan.candidate_seat else "loss"
        results.append(MatchResult(plan, outcome, record.ticks * 0.1))
    return Evaluation(TRAINING, results)


def evaluation_maps(
    generated_maps: int = DEFAULT_GENERATED_MAPS,
    bundled_maps: bool = True,
    skip_generated: int = 0,
) -> list[tuple[str, str]]:
    """The maps to evaluate on. ``skip_generated`` passes over that many generated maps first.

    Trainers pick a run's best model by its results on the first few generated
    maps. Skipping past those gives a verdict on maps that played no part in
    the choice.
    """

    maps = []
    if bundled_maps:
        for name in available_maps():
            # Maps saved from the viewer are generated, not hand-made, so they do not belong here.
            if not name.startswith("generated_") and Game(load_map(name)).players == (RED, BLUE):
                maps.append(("bundled", name))
    first = FIRST_EVALUATION_SEED + skip_generated
    maps += [("generated", str(first + index)) for index in range(generated_maps)]
    return maps


def play_match(
    definition: MapDefinition,
    red: Controller,
    blue: Controller,
    max_seconds: float = MATCH_SECONDS,
    moves: list | None = None,
) -> Game:
    """Play one match to the end. Every move made is appended to ``moves`` as (tick, seat, source, target, units)."""

    game = Game(definition)
    max_ticks = round(max_seconds / game.seconds_per_tick)
    while not game.is_finished() and game.tick < max_ticks:
        actions = {RED: red(game.view_for(RED)), BLUE: blue(game.view_for(BLUE))}
        if moves is not None:
            moves.extend(
                (game.tick, seat, action.source_id, action.target_id, action.units)
                for seat, action in actions.items()
                if action is not None
            )
        game.step(actions)
    return game


def outcome_for(game: Game, seat: int) -> str:
    if game.winner() == seat:
        return "win"
    # Out of the game is a loss even if the others have not finished yet.
    if seat not in game.living_players() or game.winner() is not None:
        return "loss"
    return "draw"


def _prepare_worker() -> None:
    import torch

    # One thread per worker; the workers themselves already use every core.
    torch.set_num_threads(1)


def _play(job: tuple) -> MatchResult:
    plan, candidate, greedy, max_seconds, (save_losses_against, save_directory) = job
    if plan.map_set == "bundled":
        definition = load_map(plan.map_key)
    else:
        definition = generate_map(int(plan.map_key)).definition

    # The same plan always gets the same seed, so a run can be repeated exactly.
    seed = zlib.crc32(repr(plan).encode())
    players = {
        plan.candidate_seat: load_player(candidate, greedy, seed),
        -plan.candidate_seat: load_player(plan.opponent, greedy, seed + 1),
    }
    moves: list = []
    game = play_match(definition, players[RED], players[BLUE], max_seconds, moves)
    outcome = outcome_for(game, plan.candidate_seat)
    saved_to = ""
    if outcome == "loss" and plan.opponent in save_losses_against:
        from ml.evaluation.saved_games import LOSSES_DIRECTORY, SavedGame

        saved = SavedGame(
            candidate, plan.opponent, plan.candidate_seat, f"{plan.map_set} {plan.map_key}", definition,
            seed, max_seconds, outcome, game.time_seconds, moves, greedy,
        )
        saved_to = str(saved.save(save_directory or LOSSES_DIRECTORY))
    return MatchResult(plan, outcome, game.time_seconds, saved_to)


def _print_progress(results: list[MatchResult], total: int, started: float) -> None:
    wins = sum(result.outcome == "win" for result in results)
    draws = sum(result.outcome == "draw" for result in results)
    line = (
        f"  {len(results):4d}/{total} games   {wins} wins, {draws} draws, "
        f"{len(results) - wins - draws} losses   {time.perf_counter() - started:4.0f}s"
    )
    if sys.stdout.isatty():
        print("\r" + line, end="\n" if len(results) == total else "", flush=True)
    elif len(results) % 25 == 0 or len(results) == total:
        print(line, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", default=UNTRAINED)
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    parser.add_argument("--generated-maps", type=int, default=DEFAULT_GENERATED_MAPS)
    parser.add_argument("--no-bundled-maps", action="store_true")
    parser.add_argument(
        "--skip-generated",
        type=int,
        default=0,
        help="Pass over this many generated maps first, for a verdict on maps no trainer has looked at",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Games per map, seat, and opponent")
    parser.add_argument("--greedy", action="store_true", help="Networks always play their most likely choice")
    parser.add_argument("--quick", action="store_true", help="Generated maps only, and just 8 of them")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-seconds", type=float, default=MATCH_SECONDS)
    parser.add_argument(
        "--save-losses-against",
        nargs="*",
        default=list(DEFAULT_SAVE_LOSSES_AGAINST),
        metavar="NAME",
        help="Save every game the candidate loses to these opponents (default: aggressive_ai; none: leave empty)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = 8 if args.quick else args.generated_maps
    bundled = not (args.quick or args.no_bundled_maps)
    maps = evaluation_maps(generated, bundled, args.skip_generated)
    bundled_count = sum(map_set == "bundled" for map_set, _ in maps)

    print(f"Candidate: {args.candidate}" + (" (greedy)" if args.greedy else ""))
    print(
        f"Maps: {bundled_count} bundled + {len(maps) - bundled_count} generated, both seats, "
        f"{args.max_seconds:g}s limit\n"
    )
    started = time.perf_counter()
    evaluation = evaluate(
        args.candidate,
        args.opponents,
        generated,
        bundled,
        args.repeats,
        args.greedy,
        args.workers,
        args.max_seconds,
        on_progress=lambda results, total: _print_progress(results, total, started),
        skip_generated=args.skip_generated,
        save_losses_against=args.save_losses_against,
    )
    print("\n" + evaluation.table())
    saved = sorted(result.saved_to for result in evaluation.results if result.saved_to)
    if saved:
        print(f"\n{len(saved)} lost games saved, to play again (python -m ml rematch FILE) or watch (python -m ml watch FILE):")
        for path in saved:
            print(f"  {path}")


if __name__ == "__main__":
    main()
