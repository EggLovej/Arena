"""Games kept for a closer look: the map, who sat where, and every move made.

The evaluator saves a game here when the candidate loses to one of the
opponents it should always beat (``--save-losses-against``). A saved game can
then be played again from the same start many times (``python -m ml rematch``)
or watched move by move in the game window (``python -m ml watch``).

One JSON file per game::

    {"candidate": "ml/runs/x/latest.pt", "opponent": "aggressive_ai",
     "candidate_seat": "red", "map_source": "generated 900123",
     "seed": 1234, "max_seconds": 180, "outcome": "loss", "seconds": 64.1, "greedy": false,
     "map": {...the map, as in maps/...},
     "moves": [[tick, "red", "source id", "target id", units], ...]}
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from game import Game
from game_types import BLUE, RED, BaseDefinition, EdgeDefinition, MapDefinition, SendAction

from ml.map_generator.generator import map_to_json


LOSSES_DIRECTORY = Path(__file__).resolve().parent.parent / "runs" / "losses"
SEAT_NAMES = {RED: "red", BLUE: "blue"}
SEATS = {name: seat for seat, name in SEAT_NAMES.items()}


@dataclass
class SavedGame:
    candidate: str
    opponent: str
    candidate_seat: int
    map_source: str  # where the map came from, for people: "generated 900123" or "bundled classic"
    map: MapDefinition
    seed: int  # the seed the candidate's chance picks were drawn with; the opponent used seed + 1
    max_seconds: float
    outcome: str  # the candidate's: "win", "draw", or "loss"
    seconds: float
    moves: list[tuple[int, int, str, str, int]]  # (tick, seat, source, target, units)
    greedy: bool = False  # networks played their most likely choice instead of picking by chance

    def save(self, directory: Path = LOSSES_DIRECTORY) -> Path:
        folder = Path(directory) / (_slug(self.candidate) + ("-greedy" if self.greedy else ""))
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{_slug(self.map_source)}_{SEAT_NAMES[self.candidate_seat]}_vs_{_slug(self.opponent)}.json"
        contents = {
            "candidate": self.candidate,
            "opponent": self.opponent,
            "candidate_seat": SEAT_NAMES[self.candidate_seat],
            "map_source": self.map_source,
            "seed": self.seed,
            "max_seconds": self.max_seconds,
            "outcome": self.outcome,
            "seconds": round(self.seconds, 1),
            "greedy": self.greedy,
            "map": json.loads(map_to_json(self.map)),
            "moves": [[tick, SEAT_NAMES[seat], source, target, units] for tick, seat, source, target, units in self.moves],
        }
        path = folder / name
        path.write_text(json.dumps(contents, indent=1) + "\n")
        return path

    def players(self) -> dict[int, str]:
        """Who sat where, by seat."""

        return {self.candidate_seat: self.candidate, -self.candidate_seat: self.opponent}

    def replay(self) -> Game:
        """The game played again from its move list; returns the finished game."""

        game = Game(self.map)
        by_tick = self.moves_by_tick()
        max_ticks = round(self.max_seconds / game.seconds_per_tick)
        while not game.is_finished() and game.tick < max_ticks:
            game.step(by_tick.get(game.tick, {}))
        return game

    def moves_by_tick(self) -> dict[int, dict[int, SendAction]]:
        by_tick: dict[int, dict[int, SendAction]] = {}
        for tick, seat, source, target, units in self.moves:
            by_tick.setdefault(tick, {})[seat] = SendAction(source, target, units)
        return by_tick


def load_game(path: str | Path) -> SavedGame:
    data = json.loads(Path(path).read_text())
    raw = data["map"]
    definition = MapDefinition(
        name=raw.get("name", "saved"),
        width=raw["width"],
        height=raw["height"],
        bases=tuple(BaseDefinition(**base) for base in raw["bases"]),
        edges=tuple(EdgeDefinition(**edge) for edge in raw["edges"]),
    )
    return SavedGame(
        candidate=data["candidate"],
        opponent=data["opponent"],
        candidate_seat=SEATS[data["candidate_seat"]],
        map_source=data["map_source"],
        map=definition,
        seed=data["seed"],
        max_seconds=data["max_seconds"],
        outcome=data["outcome"],
        seconds=data["seconds"],
        moves=[(tick, SEATS[seat], source, target, units) for tick, seat, source, target, units in data["moves"]],
        greedy=data.get("greedy", False),
    )


def short_name(player: str) -> str:
    """A player's name short enough for the game window's header.

    ``value:ml/runs/deep_from_imitation/round_0500.pt?candidates=8`` -> ``Value deep_from_imitation 500``.
    """

    kind, _, rest = player.rpartition(":") if ":" in player and not player.startswith(("untrained",)) else ("", "", player)
    path, _, options = rest.partition("?")
    if path.endswith(".pt"):
        run, _, file = path.removesuffix(".pt").rpartition("/")
        number = file.removeprefix("round_").lstrip("0") or "0" if file.startswith("round_") else file
        label = f"{run.rpartition('/')[2].replace('_', ' ')} {number}".strip()
    else:
        label = path.removesuffix("_ai").replace("_", " ").title() + (" AI" if path.endswith("_ai") else "")
    judge = next((option.removeprefix("judge=") for option in options.split("&") if option.startswith("judge=")), "")
    if judge:
        label += " + " + Path(judge).parent.name
    return f"{kind.title()} {label}" if kind else label


def saved_games(directory: Path = LOSSES_DIRECTORY) -> list[Path]:
    """Every saved game, in a fixed order, so they can be picked by number."""

    return sorted(Path(directory).glob("*/*.json"))


def find_game(text: str | None, directory: Path = LOSSES_DIRECTORY) -> Path | None:
    """A saved game by its number in ``list_games``, or by its path. ``None`` if no game was named."""

    if not text:
        return None
    if text.isdigit():
        games = saved_games(directory)
        number = int(text)
        if not 1 <= number <= len(games):
            raise SystemExit(f"There is no saved game {number}; there are {len(games)}.")
        return games[number - 1]
    return Path(text)


def list_games(directory: Path = LOSSES_DIRECTORY) -> str:
    """A numbered list of the saved games, for picking one by number."""

    games = saved_games(directory)
    if not games:
        return f"No saved games in {directory}."
    lines = [f"Saved games in {directory} (pick one by its number):"]
    folder = None
    for number, path in enumerate(games, start=1):
        if path.parent != folder:
            folder = path.parent
            lines.append(f"  {folder.name}")
        lines.append(f"  {number:4d}  {path.stem}")
    return "\n".join(lines)


def _slug(text: str) -> str:
    """A name that is safe as part of a file name: ``ml/runs/x/round_0500.pt`` -> ``ml-runs-x-round_0500.pt``."""

    return re.sub(r"[^A-Za-z0-9_.=-]+", "-", text).strip("-")
