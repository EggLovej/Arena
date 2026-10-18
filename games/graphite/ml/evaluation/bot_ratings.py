"""Ratings of the AIs in the game's menu, on our scale (strategic_ai = 1000), for the home screen.

The quick players, the scripted AIs, Neural AI and Beast, play each other
here on fresh maps. Atlas and Atlas Clean take minutes per game against a
network, so their matches come from earlier measurements (``RECORDED``, each
with the history.md section it is from), joined to the rest through players
that met both sides back then. Everything is fitted at once, the same way as
in ``ml compare``, and written to ``ais/ratings.json``, which the home screen
shows on each AI.

Neural AI is rated as the network it plays by default (``NEURAL``; long
scripts put it back into ml/runs/latest.pt when they end), at the temperature
it plays at in the game (``TEMPERATURE`` in ais/neural_ai.py), and Beast as the
player it was when this was written (``BEAST``, a ``network:`` player since
2026-10-12; the network Beast before it is ``PREVIOUS_BEAST``, the look-ahead
Beast before that ``OLD_BEAST``). The recorded matches
against Neural AI were played with its moves drawn by chance (temperature 1),
so they count for that player (``NEURAL_BY_CHANCE``), which plays the quick
players here as well and so joins them to the rest. If Beast has moved
on since, this says so: its matches then need measuring again.

    .venv/bin/python -m ml bot-ratings
"""

import argparse
import json
from datetime import date
from pathlib import Path

from ais.neural_ai import TEMPERATURE
from game_types import BLUE, RED

from ml.evaluation.compare import fit_ratings
from ml.map_generator import generate_map
from ml.models.checkpoint import load_checkpoint
from ml.play.agent import PREFIX as NETWORK, NetworkSettings
from ml.play.beast import current as current_beast
from ml.play.players import parse_name
from ml.play.rollout import Matchup, RolloutPool, Seat


RATINGS_FILE = Path(__file__).resolve().parents[2] / "ais" / "ratings.json"
REFERENCE = "strategic_ai"
QUICK = ("strategic_ai", "aggressive_ai", "careful_ai", "neural_ai", "beast_ai")
SLOW = ("atlas_ai", "atlas_clean_ai")
MENU = QUICK + SLOW
NEURAL_BY_CHANCE = "neural_ai drawing by chance"  # its network at temperature 1, as in training
PLAYED = QUICK + (NEURAL_BY_CHANCE,)  # play each other here
NEURAL = "ml/runs/deep_from_imitation/round_0500.pt"
BEAST = "network:ml/runs/free2_gap0/round_0050.pt?temperature=0.2"
PREVIOUS_BEAST = "free_gap0 round 50 at temperature 0.2 (Beast 2026-10-12 to 2026-10-13)"
OLD_BEAST = "the look-ahead Beast (deep_from_imitation round 500, judge1; Beast until 2026-10-12)"
FIRST_MAP = 980_000  # fresh maps, in the range kept for evaluation

# Players outside the menu that met Atlas, Atlas Clean or Beast and also players on our scale.
OVERNIGHT = "overnight network (64x3, round 500)"
SCORING = "move scoring (deep_scratch round 500)"
LOOKAHEAD = "look-ahead judged by the network (deep_from_imitation round 500)"

# Matches measured earlier: (first, second, the first's points, games, where in history.md).
# A win is 1 point and a draw 1/2. Atlas's forecast network was pinned to its opponent's
# network (or round 500) in all of them.
RECORDED = (
    ("beast_ai", PREVIOUS_BEAST, 121.5, 200, "A second round of the send-limit ramp (2026-10-13): 106-63, 31 draws"),
    (PREVIOUS_BEAST, OLD_BEAST, 222.0, 400, "The strongest at temperature 0.2 against Beast (2026-10-12): 213-169, 18 draws"),
    (OLD_BEAST, "atlas_ai", 49.5, 60, "The judge: 48-9, 3 draws (2026-10-02)"),
    (OLD_BEAST, NEURAL_BY_CHANCE, 39.0, 60, "The judge: 65% against the plain network (2026-10-02)"),
    (LOOKAHEAD, "strategic_ai", 54.0, 60, "Two-move look-ahead, stage 2: 53-5, 2 draws"),
    (LOOKAHEAD, "atlas_ai", 40.0, 60, "Two-move look-ahead, stage 2: 38-18, 4 draws"),
    (LOOKAHEAD, NEURAL_BY_CHANCE, 36.5, 60, "Two-move look-ahead, stage 1: 33-20, 7 draws"),
    (SCORING, "careful_ai", 39.0, 40, "Move scoring, on the deep model: 39-1"),
    (SCORING, "aggressive_ai", 38.5, 40, "Move scoring, on the deep model: 38-1, 1 draw"),
    (SCORING, "strategic_ai", 31.0, 40, "Move scoring, on the deep model: 30-8, 2 draws"),
    (SCORING, OVERNIGHT, 30.0, 40, "Move scoring, on the deep model: 29-9, 2 draws"),
    (SCORING, "atlas_ai", 24.5, 40, "Move scoring, on the deep model: 23-14, 3 draws"),
    (SCORING, "atlas_clean_ai", 24.5, 40, "Move scoring, on the deep model: 24-15, 1 draw"),
    ("atlas_ai", OVERNIGHT, 12.0, 16, "Atlas as a yardstick: 12-4"),
    ("atlas_clean_ai", "careful_ai", 30.0, 32, "Atlas Clean: 30-2"),
    ("atlas_clean_ai", "aggressive_ai", 29.0, 32, "Atlas Clean: 29-3"),
    ("atlas_clean_ai", "strategic_ai", 23.5, 32, "Atlas Clean: 23-8, 1 draw"),
    ("atlas_clean_ai", OVERNIGHT, 15.0, 32, "Atlas Clean: 15-17"),
    ("atlas_clean_ai", "atlas_ai", 5.5, 32, "Atlas Clean: 5-26, 1 draw"),
    (NEURAL_BY_CHANCE, OVERNIGHT, 71.5, 100, "Stage 10 comparison (compare_stage10_deep), imitation start round 500"),
)


def rate(games: int = 100, workers: int = 6, neural: str = NEURAL, beast: str = BEAST) -> dict:
    """Play the quick players against each other, fit those results with the recorded ones, and report."""

    if not beast.startswith(NETWORK):
        raise ValueError(f"Beast is played here as a network ({NETWORK}...), not as {beast}")
    beast_network, beast_settings = parse_name(beast.removeprefix(NETWORK), NetworkSettings)

    maps = [generate_map(FIRST_MAP + number).definition for number in range(max(games // 2, 1))]
    pairings = [(first, second) for index, first in enumerate(PLAYED) for second in PLAYED[index + 1 :]]
    matchups, owners = [], []
    for first, second in pairings:
        for definition in maps:
            matchups += [Matchup(definition, Seat(first), Seat(second)), Matchup(definition, Seat(second), Seat(first))]
            owners += [(first, second, RED), (first, second, BLUE)]
    with RolloutPool(workers) as pool:
        network = load_checkpoint(neural).model
        models = {"neural_ai": network, NEURAL_BY_CHANCE: network, "beast_ai": load_checkpoint(beast_network).model}
        temperatures = {"neural_ai": TEMPERATURE, "beast_ai": beast_settings.temperature}
        records = pool.play(matchups, models, seed=0, temperatures=temperatures)

    results: dict[tuple[str, str], list[float]] = {}
    for (first, second, seat), record in zip(owners, records):
        entry = results.setdefault((first, second), [0.0, 0])
        entry[0] += 0.5 if record.winner is None else float(record.winner == seat)
        entry[1] += 1
    for first, second, points, played, _ in RECORDED:
        entry = results.setdefault((first, second), [0.0, 0])
        entry[0] += points
        entry[1] += played

    ratings = fit_ratings(results, (REFERENCE,))
    games_of = {player: sum(count for pair, (_, count) in results.items() if player in pair) for player in ratings}
    return {
        "made": date.today().isoformat(),
        "scale": f"{REFERENCE} = 1000; 100 points more scores about 64%, 200 about 76%, 400 about 91%",
        "ratings": {name: round(ratings[name]) for name in MENU},
        "games": {name: games_of[name] for name in MENU},
        "rated as": {"neural_ai": f"{neural} at temperature {TEMPERATURE}", "beast_ai": beast},
        "others": {name: round(rating) for name, rating in ratings.items() if name not in MENU},
        "results": [[first, second, points, count] for (first, second), (points, count) in results.items()],
    }


def describe(report: dict) -> str:
    lines = [f"Ratings ({report['scale']}):"]
    for name, rating in sorted(report["ratings"].items(), key=lambda item: -item[1]):
        lines.append(f"  {name:<16} {rating:5d}   from {report['games'][name]} games")
    lines.append("Others in the fit: " + ", ".join(f"{name} {rating}" for name, rating in report["others"].items()))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=100, help="Games per pairing of the quick players (default 100)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--neural", default=NEURAL, help="The network Neural AI is rated as")
    parser.add_argument("--output", default=str(RATINGS_FILE))
    args = parser.parse_args()

    beast = current_beast()
    if beast and beast["player"] != BEAST:
        print(f"Note: Beast is now {beast['player']}; its rating here is still that of {BEAST}.", flush=True)
    report = rate(args.games, args.workers, args.neural)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(describe(report) + f"\n\nWritten to {args.output}", flush=True)


if __name__ == "__main__":
    main()
