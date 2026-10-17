"""The JSON files the website reads, written to ``web/public/data/``.

    games.json                    every game: name, tagline, seats, counts
    <game>/game.json              rules, theme, seats, maps (with their layout) and bots
    <game>/results.json           every match of the last tournament, in full: the source of the two below
    <game>/leaderboard.json       one table per match size, and how the tournament ran
    <game>/matches.json           every match of the last tournament, slimmed for the match list
    <game>/replays/<match>.json   the replays kept (one seating of every group on every map)

Each file is rewritten as a whole, so the folder always shows one complete tournament.
The leaderboard and match list are derived from results.json, so a change to how
ratings or tables are worked out is published without playing the matches again.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from arena.contract import Game
from arena.leaderboard import standings
from arena.registry import repo_root
from arena.replay import write_replay


def default_output() -> Path:
    return repo_root() / "web" / "public" / "data"


def publish_game(game: Game, output: Path) -> None:
    """game.json for one game, and its line in games.json."""

    info = game.info
    maps = []
    for map_info in game.maps():
        match = game.new_match(map_info.id, 0)
        maps.append({**asdict(map_info), "board": match.board(), "start": match.snapshot()})
    data = {
        "id": info.id,
        "name": info.name,
        "tagline": info.tagline,
        "seats": [asdict(seat) for seat in info.seats],
        "tick_seconds": info.tick_seconds,
        "max_ticks": info.max_ticks,
        "rating_anchor": list(info.rating_anchor) if info.rating_anchor else None,
        "rules": game.rules(),
        "theme": game.theme(),
        "maps": maps,
        "bots": [asdict(bot) for bot in game.bots()],
    }
    _write(output / info.id / "game.json", data)

    index_path = output / "games.json"
    games = json.loads(index_path.read_text()) if index_path.exists() else []
    games = [entry for entry in games if entry["id"] != info.id]
    games.append(
        {
            "id": info.id,
            "name": info.name,
            "tagline": info.tagline,
            "seats": [seat.id for seat in info.seats],
            "maps": len(maps),
            "bots": len(data["bots"]),
        }
    )
    _write(index_path, sorted(games, key=lambda entry: entry["id"]))


def publish_results(
    game: Game,
    output: Path,
    results: list[dict],
    replays: dict[str, dict],
    inputs: str,
    settings: dict,
    seconds: float,
) -> None:
    """The results and replays of one finished tournament, and the tables derived from them."""

    folder = output / game.info.id
    replay_folder = folder / "replays"
    if replay_folder.exists():
        shutil.rmtree(replay_folder)
    for match_id, replay in replays.items():
        write_replay(replay_folder / f"{match_id}.json", replay)
    _write(
        folder / "results.json",
        {
            "game": game.info.id,
            "updated": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "inputs": inputs,
            "fingerprint": game.fingerprint(),
            "settings": settings,
            "wall_seconds": round(seconds),
            "results": results,
        },
    )
    publish_tables(game, output)


def publish_tables(game: Game, output: Path) -> bool:
    """leaderboard.json and matches.json from the published results.json (False if there is none)."""

    folder = output / game.info.id
    path = folder / "results.json"
    if not path.exists():
        return False
    run = json.loads(path.read_text())
    results = run["results"]
    kept = {replay.stem for replay in (folder / "replays").glob("*.json")}
    _write(
        folder / "leaderboard.json",
        {
            **{key: run[key] for key in ("game", "updated", "inputs", "fingerprint", "settings")},
            "matches": len(results),
            "replays": len(kept),
            "cpu_seconds": round(sum(row["wall_seconds"] for row in results)),
            "wall_seconds": run["wall_seconds"],
            "tables": standings(game, results),
        },
    )
    _write(
        folder / "matches.json",
        [
            {
                "id": row["id"],
                "map": row["map"],
                "lineup": row["lineup"],
                "ranking": row["outcome"]["ranking"],
                "reason": row["outcome"]["reason"],
                "ticks": row["outcome"]["ticks"],
                "replay": row["id"] in kept,
            }
            for row in results
        ],
    )
    return True


def published_inputs(game: Game, output: Path) -> str | None:
    """The inputs fingerprint of the tournament currently published, if any."""

    path = output / game.info.id / "results.json"
    return json.loads(path.read_text()).get("inputs") if path.exists() else None


# Files that cannot change how bots play: notes, notebooks, tests and training scripts,
# lock files, and the shared code that only rates, checks or publishes finished matches,
# or plays matches with people on the website.
_IGNORED_SUFFIXES = (".md", ".ipynb", ".txt", ".png", ".jpg", ".lock")
_IGNORED_PARTS = ("tests", "scripts", "runs", "colab", "docs", "__pycache__", ".venv")
_IGNORED_FILES = {
    f"platform/arena/{name}"
    for name in ("__init__.py", "__main__.py", "cli.py", "leaderboard.py", "live.py", "publish.py", "ratings.py", "replay.py")
}


def inputs_fingerprint(game: Game, settings: dict) -> str:
    """A hash of everything a tournament's results depend on: the shared code, the game's files
    (rules, maps, bots, checkpoints), the packages it pins (such as torch) and the tournament
    settings. Unchanged inputs mean that playing the tournament again only reshuffles the luck
    of the neural bots."""

    root = repo_root()
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    digest.update(_dependencies(root))
    hashed = 0
    for path in _input_files(root, [Path("platform") / "arena", Path("games") / game.info.id]):
        relative = path.relative_to(root)
        if (
            path.suffix in _IGNORED_SUFFIXES
            or any(part in _IGNORED_PARTS for part in relative.parts)
            or relative.as_posix() in _IGNORED_FILES
        ):
            continue
        digest.update(relative.as_posix().encode() + b"\0" + path.read_bytes())
        hashed += 1
    if not hashed:
        raise RuntimeError(f"Found no files of {game.info.id} under {root}: is ARENA_ROOT right?")
    return digest.hexdigest()[:16]


def _dependencies(root: Path) -> bytes:
    """The packages pyproject.toml asks for, with their versions (other settings there don't count)."""

    path = root / "pyproject.toml"
    if not path.is_file():
        return b""
    project = tomllib.loads(path.read_text()).get("project", {})
    return json.dumps({key: project.get(key) for key in ("dependencies", "optional-dependencies")}, sort_keys=True).encode()


def _input_files(root: Path, folders: list[Path]) -> list[Path]:
    """Files git tracks in these folders (so local training output never counts), else every file."""

    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", *map(str, folders)],
            capture_output=True,
            check=True,
        ).stdout.decode()
        files = [root / name for name in listed.split("\0") if name]
        if files:
            return sorted(path for path in files if path.is_file())
    except (OSError, subprocess.CalledProcessError):
        pass
    found = []
    for folder in folders:
        for directory, subdirectories, names in os.walk(root / folder):
            subdirectories[:] = sorted(name for name in subdirectories if name not in _IGNORED_PARTS)
            found += [Path(directory) / name for name in names]
    return sorted(found)


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n")
