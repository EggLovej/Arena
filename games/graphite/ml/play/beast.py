"""Beast: whichever player is currently our strongest, playable in the game as "Beast AI".

The choice is one small file, ``ml/runs/beast.json``: the player's name (any
name the evaluator understands: a checkpoint, ``value:``, ``lookahead:``,
or an AI file), when it became Beast, and why. Every time an
evaluation shows a new strongest player, Beast is moved to it:

    .venv/bin/python -m ml beast                      show who Beast is
    .venv/bin/python -m ml beast set "lookahead:ml/runs/x/latest.pt?judge=ml/runs/j/best.pt" \\
        --why "beat the old Beast 60% over 100 games"

``ais/beast_ai.py`` reads the file at the start of every match.
"""

import argparse
import json
from datetime import date
from pathlib import Path


BEAST_FILE = Path(__file__).resolve().parent.parent / "runs" / "beast.json"


def current(path: Path | None = None) -> dict | None:
    """Beast's entry: {"player", "since", "why", "before"}; None if no Beast was chosen yet."""

    path = path or BEAST_FILE
    return json.loads(path.read_text()) if path.exists() else None


def crown(player: str, why: str, path: Path | None = None) -> dict:
    """Make ``player`` the new Beast; the previous ones are kept in "before"."""

    from ml.evaluation.evaluate import load_player

    path = path or BEAST_FILE
    load_player(player)  # refuse a name that cannot be played
    previous = current(path)
    history = []
    if previous:
        history = previous.get("before", []) + [{key: previous[key] for key in ("player", "since", "why")}]
    entry = {"player": player, "since": date.today().isoformat(), "why": why, "before": history}
    path.write_text(json.dumps(entry, indent=1) + "\n")
    return entry


def describe(entry: dict | None) -> str:
    if entry is None:
        return f"No Beast chosen yet ({BEAST_FILE} is missing)."
    lines = [f"Beast since {entry['since']}: {entry['player']}", f"  why: {entry['why']}"]
    for old in reversed(entry.get("before", [])):
        lines.append(f"  before ({old['since']}): {old['player']}")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command")
    choose = commands.add_parser("set", help="Make a player the new Beast")
    choose.add_argument("player", help="Any player name the evaluator understands")
    choose.add_argument("--why", required=True, help="The evidence, e.g. the deciding match result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "set":
        crown(args.player, args.why)
    print(describe(current()))


if __name__ == "__main__":
    main()
