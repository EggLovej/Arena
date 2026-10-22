"""Every number in Blast, and its maps.

Balancing the game is a change to this file, and to the numbers ``RULES.md``
quotes. Times are in ticks of a twentieth of a second, and every value is a
whole number, so a match plays out exactly the same anywhere.

A map is a file in ``maps/``: its name, a line about it, how it mirrors (see
``Layout``) and its rows, where ``#`` is a wall, ``.`` a floor tile that may
start with a crate, ``-`` one kept clear and ``1`` to ``4`` the seats' starts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

TICKS_PER_SECOND = 20
FUSE = 50  # ticks from dropping a bomb to its blast
FLAME = 10  # ticks a blast's flames burn
STEPS = (6, 5, 4, 3)  # ticks a step from one tile to the next takes, at each speed
START_BOMBS = 1  # bombs a bomber can have on the field at once
START_RANGE = 2  # tiles a blast reaches in each direction
MAX_BOMBS = 8
MAX_RANGE = 8
CLOSING = 90 * TICKS_PER_SECOND  # when the walls start closing in
CLOSE_EVERY = 6  # ticks between two tiles of the closing walls
MAX_TICKS = 3 * 60 * TICKS_PER_SECOND  # never reached: by then the walls have filled every map

CRATES = 70  # percent of the free floor that starts with a crate
POWERS = {"bomb": 16, "fire": 16, "speed": 8}  # what a crate hides, out of 100; the rest hide nothing
MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

WALL, FLOOR = "#", "."
SEATS = ("red", "blue", "green", "yellow")  # starts 1 to 4
MAPS = Path(__file__).resolve().parent.parent / "maps"


@dataclass(frozen=True)
class Layout:
    """A map. Crates are dealt anew for every match, the same way for tiles that ``mirror`` ("x",
    "y" or "xy") maps onto each other, so that starts that mirror each other have the same around
    them."""

    id: str
    name: str
    description: str
    mirror: str
    rows: tuple[str, ...]

    @property
    def seats(self) -> tuple[str, ...]:
        return tuple(seat for number, seat in enumerate(SEATS, 1) if any(str(number) in row for row in self.rows))


@cache
def layouts() -> dict[str, Layout]:
    """Every map in ``maps/``, by id: its file's name."""

    found = {}
    for path in sorted(MAPS.glob("*.json")):
        data = json.loads(path.read_text())
        found[path.stem] = Layout(path.stem, data["name"], data["description"], data["mirror"], tuple(data["rows"]))
    return found
