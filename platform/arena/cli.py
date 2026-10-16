"""The ``arena`` command line.

    python -m arena games [--ids]
    python -m arena match graphite --map classic --bots strategic_ai careful_ai --replay match.json
    python -m arena tournament graphite --workers 4
    python -m arena verify web/public/data/graphite/replays/*.json
    python -m arena publish graphite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from arena.contract import Outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arena", description="Play, rate and publish bot matches.")
    commands = parser.add_subparsers(dest="command", required=True)

    games = commands.add_parser("games", help="List the games and their bots and maps")
    games.add_argument("--ids", action="store_true", help="Only the game ids, one a line (for scripts)")

    match = commands.add_parser("match", help="Play one match")
    match.add_argument("game")
    match.add_argument("--map", required=True)
    match.add_argument("--bots", nargs="+", required=True, help="One bot per seat, in seat order")
    match.add_argument("--seed", type=int, default=0)
    match.add_argument("--replay", type=Path, help="Write the replay to this file")

    tournament = commands.add_parser("tournament", help="Play every pairing on every map and publish the results")
    tournament.add_argument("game")
    tournament.add_argument("--bots", nargs="+", help="Only these bots (default: every leaderboard bot)")
    tournament.add_argument("--maps", nargs="+", help="Only these maps (default: all)")
    tournament.add_argument("--rounds", type=int, default=1, help="Times each seating is played (default 1)")
    tournament.add_argument("--workers", type=int, default=1)
    tournament.add_argument("--output", type=Path, help="Data folder (default: web/public/data)")
    tournament.add_argument("--force", action="store_true", help="Play even if nothing changed since the last tournament")

    verify = commands.add_parser("verify", help="Check that replays play back exactly as recorded")
    verify.add_argument("replays", nargs="+", type=Path)

    publish = commands.add_parser(
        "publish", help="Rewrite a game's website files (maps, bots, rules, tables) without playing any matches"
    )
    publish.add_argument("game")
    publish.add_argument("--output", type=Path, help="Data folder (default: web/public/data)")

    args = parser.parse_args(argv)
    # Loading a game can change the working directory, so fix every path first.
    for name in ("replay", "output"):
        if getattr(args, name, None) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if getattr(args, "replays", None):
        args.replays = [path.resolve() for path in args.replays]
    return {"games": _games, "match": _match, "tournament": _tournament, "verify": _verify, "publish": _publish}[
        args.command
    ](args)


def _games(args) -> int:
    from arena.registry import available_games, load_game

    if args.ids:
        print("\n".join(available_games()))
        return 0
    for game_id in available_games():
        game = load_game(game_id)
        print(f"{game.info.name} ({game_id}): {game.info.tagline}")
        print("  maps: " + ", ".join(f"{m.id} ({len(m.seats)}p)" for m in game.maps()))
        print("  bots: " + ", ".join(f"{b.id} ({b.kind})" for b in game.bots()))
    return 0


def _match(args) -> int:
    from arena.registry import load_game
    from arena.replay import write_replay
    from arena.runner import play_match

    game = load_game(args.game)
    seats = next((m.seats for m in game.maps() if m.id == args.map), None)
    if seats is None:
        print(f"No map {args.map!r}", file=sys.stderr)
        return 2
    if len(args.bots) != len(seats):
        print(f"{args.map} has {len(seats)} seats ({', '.join(seats)}); name one bot for each", file=sys.stderr)
        return 2
    record = play_match(game, args.map, dict(zip(seats, args.bots)), args.seed, record_replay=args.replay is not None)
    print(_describe(record.lineup, record.outcome))
    for seat, report in record.seats.items():
        line = report.to_json()
        print(f"  {seat}: {line['bot']}, {line['think_ms_mean']:.2f} ms a move, {line['errors']} errors, {line['refused']} refused")
    if args.replay:
        write_replay(args.replay, record.replay)
        print(f"Replay written to {args.replay}")
    return 0


def _tournament(args) -> int:
    from arena import publish, tournament
    from arena.registry import load_game
    from arena.replay import verify

    game = load_game(args.game)
    output = args.output or publish.default_output()
    settings = {"bots": sorted(args.bots) if args.bots else None, "maps": sorted(args.maps) if args.maps else None, "rounds": args.rounds}
    inputs = publish.inputs_fingerprint(game, settings)
    if not args.force and publish.published_inputs(game, output) == inputs:
        print(f"Nothing changed since the published tournament (inputs {inputs}); use --force to play it again.")
        return 0

    plans = tournament.schedule(game, args.bots, args.maps, args.rounds)
    print(f"{len(plans)} matches with {args.workers} worker(s); inputs {inputs}", flush=True)
    started = time.perf_counter()
    order = {plan.match_id: number for number, plan in enumerate(plans)}
    results, replays = [], {}
    for done, (plan, row, replay) in enumerate(tournament.run(game.info.id, plans, args.workers), start=1):
        results.append(row)
        if replay is not None:
            replays[plan.match_id] = replay
        elapsed = time.perf_counter() - started
        print(f"[{done}/{len(plans)} {elapsed:6.0f}s] {_describe(row['lineup'], Outcome.from_json(row['outcome']))} on {row['map']} ({row['wall_seconds']:.1f}s)", flush=True)
    results.sort(key=lambda row: order[row["id"]])

    problems = {match_id: verify(game, replay) for match_id, replay in replays.items()}
    problems = {match_id: found for match_id, found in problems.items() if found}
    if problems:
        for match_id, found in problems.items():
            print(f"Replay {match_id} does not play back: {'; '.join(found)}", file=sys.stderr)
        return 1

    publish.publish_game(game, output)
    publish.publish_results(game, output, results, replays, inputs, settings, time.perf_counter() - started)
    errors = sum(seat["errors"] for row in results for seat in row["seats"].values())
    print(f"Published {len(results)} matches and {len(replays)} replays to {output} ({errors} bot errors)")
    return 0


def _verify(args) -> int:
    from arena.registry import load_game
    from arena.replay import read_replay, verify

    failed = 0
    for path in args.replays:
        replay = read_replay(path)
        problems = verify(load_game(replay["game"]), replay)
        if problems:
            failed += 1
            print(f"{path.name}: {'; '.join(problems)}")
    print(f"{len(args.replays) - failed} of {len(args.replays)} replays play back exactly")
    return 1 if failed else 0


def _publish(args) -> int:
    from arena import publish
    from arena.registry import load_game

    output = args.output or publish.default_output()
    game = load_game(args.game)
    publish.publish_game(game, output)
    if publish.publish_tables(game, output):
        print(f"Wrote {output / args.game}: game.json, leaderboard.json and matches.json")
    else:
        print(f"Wrote {output / args.game / 'game.json'} (no tournament published yet)")
    return 0


def _describe(lineup: dict[str, str], outcome: Outcome) -> str:
    places = " > ".join(" = ".join(lineup[seat] for seat in group) for group in outcome.ranking)
    return f"{places} ({outcome.reason}, tick {outcome.ticks})"


if __name__ == "__main__":
    sys.exit(main())
