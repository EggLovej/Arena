"""Watch a saved game in the game window, move by move as it was played.

    .venv/bin/python -m ml watch                       list the saved games, numbered
    .venv/bin/python -m ml watch 3                     watch saved game 3
    .venv/bin/python -m ml watch 3 --speed 3           three times as fast
    .venv/bin/python -m ml watch 3 --live              the same players play it again, live
    .venv/bin/python -m ml watch path/to/game.json     a saved game by its file

By default the window shows exactly the recorded moves, so it is the very
game that was lost. ``--live`` lets the players decide again (with the seeds
of the saved game), which shows the same game for players that do not pick
by chance, and a fresh one for those that do.
"""

import argparse

from game import Game
from game_types import BLUE, RED

from ml.evaluation.evaluate import load_player
from ml.evaluation.saved_games import SEAT_NAMES, SavedGame, find_game, list_games, load_game, short_name


def recorded_players(saved: SavedGame) -> dict:
    """One controller per seat that plays back the recorded moves."""

    by_tick = saved.moves_by_tick()

    def player(state):
        return by_tick.get(state.tick, {}).get(state.player_id)

    return {RED: player, BLUE: player}


def live_players(saved: SavedGame) -> dict:
    return {
        saved.candidate_seat: load_player(saved.candidate, saved.greedy, saved.seed),
        -saved.candidate_seat: load_player(saved.opponent, saved.greedy, saved.seed + 1),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game", nargs="?", help="A saved game's number (leave out to list them) or its file")
    parser.add_argument("--speed", type=float, default=1.0, help="Game seconds per real second (default 1)")
    parser.add_argument("--live", action="store_true", help="Let the players play again instead of replaying the record")
    return parser.parse_args()


def main() -> None:
    from main import play_visual

    args = parse_args()
    path = find_game(args.game)
    if path is None:
        print(list_games())
        return
    saved = load_game(path)
    players = live_players(saved) if args.live else recorded_players(saved)
    for seat in (RED, BLUE):
        print(f"{SEAT_NAMES[seat]:5}: {saved.players()[seat]}")
    print(f"Map: {saved.map_source}. Saved outcome for the candidate: {saved.outcome} after {saved.seconds:.0f}s")
    game = Game(saved.map)
    max_ticks = round(saved.max_seconds / game.seconds_per_tick)
    names = {seat: short_name(player) for seat, player in saved.players().items()}
    play_visual(game, players[RED], players[BLUE], max_ticks, args.speed, player_names=names)


if __name__ == "__main__":
    main()
