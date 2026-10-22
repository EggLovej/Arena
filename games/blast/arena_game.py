"""Blast as the Arena platform sees it: the adapter between this folder and ``platform/arena``.

The rules are the ``blast`` package in this folder, the maps are in ``maps/``, the
board's themes in ``themes/`` and the bots in ``ais/``. Loading this adapter puts
this folder on ``sys.path`` so that all of them can import ``blast``. Bots are
loaded from their files by path, because Graphite's adapter puts its own ``ais``
package on the path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blast.field import ACTIONS, Field  # noqa: E402
from blast.rules import FUSE, MAX_TICKS, MOVES, SEATS, TICKS_PER_SECOND, layouts  # noqa: E402

from arena.contract import BotInfo, GameInfo, MapInfo, Outcome, SeatInfo, bot_from_module, summary  # noqa: E402

# The default theme, which gives the seats their colours; the others in themes/ follow it.
THEME_FILE = ROOT / "themes" / "garden.json"
THEME = json.loads(THEME_FILE.read_text())

INFO = GameInfo(
    id="blast",
    name="Blast",
    tagline="Bombs on a grid for up to four: break crates, grab power-ups, and be the last one standing.",
    seats=tuple(SeatInfo(seat, seat.title(), THEME["players"][seat]["fill"]) for seat in SEATS),
    tick_seconds=1 / TICKS_PER_SECOND,
    max_ticks=MAX_TICKS,
    rating_anchor=("digger_ai", 1000.0),
)


class BlastMatch:
    def __init__(self, map_id: str, seed: int):
        self.field = Field(layouts()[map_id], seed)
        self.seats = tuple(self.field.bombers)

    @property
    def tick(self) -> int:
        return self.field.tick

    def acting(self) -> tuple[str, ...]:
        return tuple(bomber.seat for bomber in self.field.alive())

    def observe(self, seat: str):
        return self.field.observe(seat)

    def step(self, actions: dict[str, str | None]) -> dict[str, str]:
        return self.field.step(actions)

    def finished(self) -> bool:
        return self.field.finished()

    def outcome(self) -> Outcome:
        """Whoever is left first (sharing first place if time ran out), then the rest, the last
        caught first; bombers caught in the same tick share their place."""

        bombers = self.field.bombers.values()
        left = tuple(bomber.seat for bomber in bombers if bomber.out < 0)
        ranking = [left] if left else []
        for tick in sorted({bomber.out for bomber in bombers if bomber.out >= 0}, reverse=True):
            ranking.append(tuple(bomber.seat for bomber in bombers if bomber.out == tick))
        return Outcome(tuple(ranking), "elimination" if self.finished() else "time", self.tick)

    def board(self) -> dict:
        """The map's rows, how long a fuse burns, and when each tile the walls close in on closes."""

        return {
            "name": self.field.layout.name,
            "rows": list(self.field.layout.rows),
            "fuse": FUSE,
            "closes": [[*tile, tick] for tile, tick in self.field.closes.items()],
        }

    def snapshot(self) -> dict:
        return self.field.snapshot()

    def save_state(self) -> dict:
        return self.field.save_state()


class WebPerson:
    """A person playing on the website. The page sends the ways they hold, whenever they change,
    and each press of the bomb key:

    - ``{"walk": ["up", "left"]}``: whenever the bomber can step, step the first of these ways
      (up, down, left or right) that is open, until the next walk command; ``[]`` stands still
    - ``"bomb"``: drop a bomb on the next tick
    """

    def __init__(self):
        self.walk: list[str] = []
        self.bomb = False

    def command(self, data) -> None:
        if data == "bomb":
            self.bomb = True
        elif isinstance(data, dict) and data.keys() == {"walk"} and isinstance(data["walk"], list) and set(data["walk"]) <= MOVES.keys():
            self.walk = data["walk"]
        else:
            raise ValueError(f"not a command: {data!r}")

    def act(self, observation) -> str | None:
        if self.bomb:
            self.bomb = False
            return "bomb"
        if observation.me.ready:
            return None
        # Walking into a wall does nothing, rather than saying each tick that the step was refused.
        x, y = observation.me.tile
        return next((way for way in self.walk if observation.open((x + MOVES[way][0], y + MOVES[way][1]))), None)

    def view(self) -> dict:
        return {}


class BlastGame:
    info = INFO

    def maps(self) -> tuple[MapInfo, ...]:
        return tuple(MapInfo(layout.id, layout.name, layout.seats, layout.description) for layout in layouts().values())

    def bots(self) -> tuple[BotInfo, ...]:
        return tuple(
            BotInfo(id=name, name=name.removesuffix("_ai").title(), description=summary(ROOT / "ais" / f"{name}.py"))
            for name in _roster()
        )

    def load_bot(self, bot_id: str):
        if bot_id not in _roster():
            raise ValueError(f"Blast has no bot named {bot_id!r}")
        spec = importlib.util.spec_from_file_location(f"blast_ais_{bot_id}", ROOT / "ais" / f"{bot_id}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return bot_from_module(module)

    def new_match(self, map_id: str, seed: int) -> BlastMatch:
        return BlastMatch(map_id, seed)

    def action_to_json(self, action) -> str:
        """An action is its name, one of ``blast.ACTIONS``."""

        if not isinstance(action, str):
            raise TypeError(f"an action is one of {', '.join(ACTIONS)}, not {type(action).__name__}")
        if action not in ACTIONS:
            raise ValueError(f"{action!r} is not one of {', '.join(ACTIONS)}")
        return action

    def action_from_json(self, data: str) -> str:
        return data

    def person(self, seat: str) -> WebPerson:
        return WebPerson()

    def themes(self) -> tuple[dict, ...]:
        others = sorted(path for path in THEME_FILE.parent.glob("*.json") if path != THEME_FILE)
        return (THEME, *(json.loads(path.read_text()) for path in others))


def _roster() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in (ROOT / "ais").glob("*.py") if not path.stem.startswith("_")))


GAME = BlastGame()
