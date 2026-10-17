"""Evaluate one AI against an opponent on every installed map and every seat."""

import argparse

from game import Game
from game_types import BLUE, GREEN, RED
from loading import available_maps, load_ai, load_map
from main import play_headless, player_name


def evaluate(candidate_name: str, opponent_name: str, tick_seconds: float, max_seconds: float):
    candidate = load_ai(candidate_name)
    opponent = load_ai(opponent_name)
    max_ticks = int(max_seconds / tick_seconds)
    results = []

    for map_name in available_maps():
        map_definition = load_map(map_name)
        active_players = Game(map_definition).players
        for candidate_color in active_players:
            game = Game(load_map(map_name), seconds_per_tick=tick_seconds)
            controllers = {
                RED: candidate if candidate_color == RED else opponent,
                BLUE: candidate if candidate_color == BLUE else opponent,
                GREEN: candidate if candidate_color == GREEN else opponent,
            }
            winner = play_headless(
                game,
                controllers[RED],
                controllers[BLUE],
                max_ticks,
                controllers[GREEN] if GREEN in active_players else None,
            )

            if winner == candidate_color:
                outcome = "win"
                score = 1.0
            elif winner is None and candidate_color in game.living_players():
                outcome = "draw"
                score = 0.5
            else:
                outcome = "loss"
                score = 0.0

            results.append(score)
            print(
                f"{map_name:14} candidate={player_name(candidate_color):5} "
                f"result={outcome:4} ticks={game.tick}"
            )

    total = sum(results)
    print(f"\nScore: {total:g}/{len(results)}")
    return total, len(results)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--opponent", default="careful_ai")
    parser.add_argument("--tick-seconds", type=float, default=0.1)
    parser.add_argument("--max-seconds", type=float, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tick_seconds <= 0 or args.max_seconds <= 0:
        raise ValueError("tick-seconds and max-seconds must be positive")
    evaluate(args.candidate, args.opponent, args.tick_seconds, args.max_seconds)


if __name__ == "__main__":
    main()
