"""Epoch as the Arena platform sees it: the adapter between this folder and ``platform/arena``.

The rules are the ``epoch`` package in this folder and the bots are in ``ais/``;
loading this adapter puts this folder on ``sys.path`` so that both can import
``epoch``. Bots are loaded from their files by path, so another game's ``ais``
can't stand in for them.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from epoch.battle import ACTIONS, Battle  # noqa: E402
from epoch.rules import AGES, BASE_LENGTH, LANES, MAX_TICKS, PLACES, QUEUE, TICKS_PER_SECOND, TURRETS, UNITS  # noqa: E402

from arena.contract import BotInfo, GameInfo, MapInfo, Outcome, SeatInfo, bot_from_module, summary  # noqa: E402

SEATS = ("red", "blue")  # the left base and the right one

INFO = GameInfo(
    id="epoch",
    name="Epoch",
    tagline="Two bases, one lane: train an army, hold the line, and race your enemy through the ages.",
    seats=(SeatInfo("red", "Red", "#c8402f"), SeatInfo("blue", "Blue", "#2f6fd0")),
    tick_seconds=1 / TICKS_PER_SECOND,
    max_ticks=MAX_TICKS,
    rating_anchor=("balanced_ai", 1000.0),
)


class EpochMatch:
    def __init__(self, map_id: str):
        self.lane = LANES[map_id]
        self.battle = Battle(self.lane.length, SEATS)
        self.seats = SEATS

    @property
    def tick(self) -> int:
        return self.battle.tick

    def acting(self) -> tuple[str, ...]:
        return self.seats

    def observe(self, seat: str):
        return self.battle.observe(seat)

    def step(self, actions: dict[str, str | None]) -> dict[str, str]:
        return self.battle.step(actions)

    def finished(self) -> bool:
        return self.battle.finished()

    def outcome(self) -> Outcome:
        """A destroyed base loses, and two destroyed at once draw. When time runs out, the base
        with more hit points left wins."""

        red, blue = (self.battle.side(seat).hp for seat in SEATS)
        reason = "elimination" if self.finished() else "time"
        if red == blue or (red <= 0 and blue <= 0):
            return Outcome((SEATS,), reason, self.tick)
        ranking = SEATS if red > blue else SEATS[::-1]
        return Outcome(tuple((seat,) for seat in ranking), reason, self.tick)

    def board(self) -> dict:
        """The lane, and every age, unit and turret with the numbers the website draws and prices."""

        return {
            "name": self.lane.name,
            "length": self.lane.length,
            "base_length": BASE_LENGTH,
            "queue": QUEUE,
            "places": list(PLACES),
            "ages": [
                {
                    "id": age.id,
                    "name": age.name,
                    "xp": age.xp,
                    "price": age.price,
                    "base_hp": age.base_hp,
                    "units": [unit.id for unit in age.units],
                    "turret": age.turret.id,
                }
                for age in AGES
            ],
            "units": {
                unit.id: {key: getattr(unit, key) for key in ("name", "role", "price", "train", "hp", "reach", "speed", "length")}
                for unit in UNITS.values()
            },
            "turrets": {turret.id: {"name": turret.name, "price": turret.price, "reach": turret.reach} for turret in TURRETS.values()},
        }

    def snapshot(self) -> dict:
        return self.battle.snapshot()

    def save_state(self) -> dict:
        return self.battle.save_state()


class WebPerson:
    """A person on the website: each button or key they press is one action, played in turn, one a tick."""

    def __init__(self):
        self.pressed: deque[str] = deque()

    def command(self, data) -> None:
        if data not in ACTIONS:
            raise ValueError(f"{data!r} is not one of {', '.join(ACTIONS)}")
        self.pressed.append(data)

    def act(self, observation) -> str | None:
        return self.pressed.popleft() if self.pressed else None

    def view(self) -> dict:
        return {}


class EpochGame:
    info = INFO

    def maps(self) -> tuple[MapInfo, ...]:
        return tuple(MapInfo(lane.id, lane.name, SEATS, lane.description) for lane in LANES.values())

    def bots(self) -> tuple[BotInfo, ...]:
        return tuple(
            BotInfo(id=name, name=name.removesuffix("_ai").title(), description=summary(ROOT / "ais" / f"{name}.py"))
            for name in _roster()
        )

    def load_bot(self, bot_id: str):
        if bot_id not in _roster():
            raise ValueError(f"Epoch has no bot named {bot_id!r}")
        spec = importlib.util.spec_from_file_location(f"epoch_ais_{bot_id}", ROOT / "ais" / f"{bot_id}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return bot_from_module(module)

    def new_match(self, map_id: str, seed: int) -> EpochMatch:
        # Epoch's rules have no chance in them, so the seed only matters to bots.
        return EpochMatch(map_id)

    def action_to_json(self, action) -> str:
        """An action is its name, one of ``epoch.ACTIONS``."""

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
        """One look for now: the artwork's. Besides its sprites and lane backdrop, the board draws
        with the sky above the backdrop, the art's ink and the cream of its panels."""

        return (
            {
                "id": "daylight",
                "name": "Daylight",
                "description": "The lane on a sunny day, as the artwork draws it.",
                "sky": "#9ccfdf",
                "ink": "#2b211c",
                "panel": "#fff8e8",
                "players": {seat.id: seat.color for seat in INFO.seats},
            },
        )


def _roster() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in (ROOT / "ais").glob("*.py") if not path.stem.startswith("_")))


GAME = EpochGame()
