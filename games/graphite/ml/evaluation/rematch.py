"""Play a saved game's start again many times, from both seats, and count the results.

Was a lost game bad luck, a map that favours one seat, or a real weakness?
This plays the same map with the same two players again: ``--games`` times
with the candidate in the seat it lost from, and as often from the other
seat. Each game draws the players' chance picks from a different seed, so a
player that picks by chance plays a different game each time; a player that
always plays its best move (the scripted AIs, move scoring) plays the same
game every time from the same seat.

    .venv/bin/python -m ml rematch                     list the saved games, numbered
    .venv/bin/python -m ml rematch 3                   play saved game 3's start again, 50 times per seat
    .venv/bin/python -m ml rematch 3 --games 20 --candidate value:ml/runs/latest.pt
"""

import argparse
import os
import statistics
from multiprocessing import Pool

from game_types import BLUE, RED

from ml.evaluation.evaluate import _prepare_worker, load_player, outcome_for, play_match
from ml.evaluation.saved_games import SEAT_NAMES, SavedGame, find_game, list_games, load_game


def rematch(saved: SavedGame, games: int = 50, candidate: str | None = None, workers: int | None = None) -> dict:
    """Results per seat of the candidate: {"red": [(outcome, seconds, moves made), ...], "blue": [...]}.

    ``candidate`` plays instead of the saved game's candidate, for example a newer model.
    """

    candidate = candidate or saved.candidate
    jobs = [
        (saved, candidate, seat, saved.seed + 1000 * number)
        for seat in (saved.candidate_seat, -saved.candidate_seat)
        for number in range(games)
    ]
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 2))
    if workers == 1:
        played = list(map(_play_again, jobs))
    else:
        with Pool(workers, initializer=_prepare_worker) as pool:
            played = pool.map(_play_again, jobs)
    results = {SEAT_NAMES[saved.candidate_seat]: [], SEAT_NAMES[-saved.candidate_seat]: []}
    for (_, _, seat, _), result in zip(jobs, played):
        results[SEAT_NAMES[seat]].append(result)
    return results


def describe(saved: SavedGame, results: dict, candidate: str | None = None) -> str:
    candidate = candidate or saved.candidate
    lines = [
        f"Map: {saved.map_source} ({len(saved.map.bases)} bases)",
        f"Candidate: {candidate}" + (" (greedy)" if saved.greedy else ""),
        f"Opponent: {saved.opponent}",
        f"Saved game: the candidate played {SEAT_NAMES[saved.candidate_seat]} and lost after {saved.seconds:.0f}s",
        "",
        f"{'candidate plays':16}{'games':>6}{'wins':>6}{'draws':>7}{'losses':>8}{'score':>8}"
        f"{'median length':>15}{'different games':>17}",
    ]
    for seat_name, played in results.items():
        wins = sum(outcome == "win" for outcome, _, _ in played)
        draws = sum(outcome == "draw" for outcome, _, _ in played)
        losses = len(played) - wins - draws
        score = (wins + draws / 2) / len(played)
        median = statistics.median(seconds for _, seconds, _ in played)
        # Games with the same length and the same number of moves are almost surely the same game.
        different = len({(round(seconds, 1), moves) for _, seconds, moves in played})
        lines.append(
            f"{seat_name:16}{len(played):6d}{wins:6d}{draws:7d}{losses:8d}{score:8.0%}{median:14.0f}s{different:17d}"
        )
    return "\n".join(lines)


def _play_again(job) -> tuple[str, float, int]:
    outcome, seconds, moves, _ = play_again(*job)
    return outcome, seconds, len(moves)


def play_again(saved: SavedGame, candidate: str, seat: int, seed: int, save_loss: bool = False):
    """Play a saved game's start once more: (outcome, seconds, moves, file of the saved loss or "").

    With ``save_loss``, a lost game is saved like the evaluator saves one, so it can be watched.
    """

    players = {
        seat: load_player(candidate, saved.greedy, seed),
        -seat: load_player(saved.opponent, saved.greedy, seed + 1),
    }
    moves: list = []
    game = play_match(saved.map, players[RED], players[BLUE], saved.max_seconds, moves)
    outcome = outcome_for(game, seat)
    path = ""
    if save_loss and outcome == "loss":
        lost = SavedGame(
            candidate, saved.opponent, seat, saved.map_source, saved.map, seed, saved.max_seconds,
            outcome, game.time_seconds, moves, saved.greedy,
        )
        path = str(lost.save())
    return outcome, game.time_seconds, moves, path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game", nargs="?", help="A saved game's number (leave out to list them) or its file")
    parser.add_argument("--games", type=int, default=50, help="Games from each seat (default 50)")
    parser.add_argument("--candidate", help="Play this player instead of the saved game's candidate")
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = find_game(args.game)
    if path is None:
        print(list_games())
        return
    saved = load_game(path)
    results = rematch(saved, args.games, args.candidate, args.workers)
    print(describe(saved, results, args.candidate))


if __name__ == "__main__":
    main()
