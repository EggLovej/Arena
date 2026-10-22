"""Splice as the Arena platform sees it: the adapter between this folder and ``platform/arena``.

The rules are the ``splice`` package in this folder and the bots are in ``ais/``;
loading this adapter puts this folder on ``sys.path`` so that both can import
``splice``. Bots are loaded from their files by path, so another game's ``ais``
can't stand in for them.

Splice is turn-based: each tick, only the player whose turn it is acts. A tick
without a move (a person still thinking) leaves the turn where it is, so the
match clock runs on while they think.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from splice import MAPS, Move, cells_in  # noqa: E402

from arena.contract import BotInfo, GameInfo, MapInfo, Outcome, SeatInfo, bot_from_module, summary  # noqa: E402

THEMES = ROOT / "themes"
THEME = json.loads((THEMES / "petri_dish.json").read_text())  # the default, which gives the seats their colours
SEATS = ("magenta", "cyan", "amber")  # in turn order; a map's first player is Magenta
NAMES = {"mcts_ai": "Monte Carlo"}  # the rest are called by their file names: greedy_ai is Greedy

INFO = GameInfo(
    id="splice",
    name="Splice",
    tagline="Turn-based territory on hexagons: grow or jump, and win over every neighbour where you land.",
    seats=tuple(SeatInfo(seat, seat.title(), THEME["players"][seat]["fill"]) for seat in SEATS),
    tick_seconds=0.5,
    max_ticks=3600,  # half an hour: bots finish in a few minutes, people may take longer
    rating_anchor=("greedy_ai", 1000.0),
)


class SpliceMatch:
    def __init__(self, map_id: str):
        self.position = MAPS[map_id].start
        self.seats = SEATS[: len(self.position.pieces)]
        self.tick = 0
        self.last: list | None = None  # the last move, as snapshot() shows it

    def acting(self) -> tuple[str, ...]:
        return () if self.position.turn is None else (self.seats[self.position.turn],)

    def observe(self, seat: str):
        """The ``splice.Position``, which says whose turn it is: the seat's own, when it is asked."""

        return self.position

    def step(self, actions: dict[str, Move | None]) -> dict[str, str]:
        self.tick += 1
        if self.position.turn is None:
            return {}
        seat = self.seats[self.position.turn]
        move = actions.get(seat)
        if move is None:
            return {}
        refusal = self.position.refusal(move)
        if refusal:
            return {seat: refusal}
        before = self.position
        self.position = before.play(move)
        owners = dict(self._owners(before))
        changed = [(cell, owner) for cell, owner in self._owners(self.position) if owners.get(cell) != owner]
        won = [[cell, owners[cell]] for cell, _ in changed if cell in owners]
        filled = [cell for cell, _ in changed if cell not in owners and cell != move[1]]
        self.last = [seat, *move, self.tick, won, filled]
        return {}

    def finished(self) -> bool:
        return self.position.turn is None

    def outcome(self) -> Outcome:
        """Most pieces first; players with as many share their place. A match also ends this way
        on time. The reason says how it ended: one player left ("elimination"), nobody else able
        to move ("filled") or the clock ("time")."""

        counts = dict(zip(self.seats, self.position.counts()))
        ranking = tuple(tuple(seat for seat in self.seats if counts[seat] == count) for count in sorted(set(counts.values()), reverse=True))
        if not self.finished():
            reason = "time"
        else:
            reason = "elimination" if sum(1 for count in counts.values() if count) == 1 else "filled"
        return Outcome(ranking, reason, self.tick)

    def board(self) -> dict:
        """The cells as [column, row]: a row's cells are two columns apart, and rows are shifted a column."""

        return {"cells": [list(cell) for cell in self.position.board.cells]}

    def snapshot(self) -> dict:
        """Each piece's owner by cell; whose turn it is; and the last move, as [seat, from, to,
        tick, [[cell, previous owner] for each piece it won], [cells the end of the game filled]]."""

        return {
            "pieces": {str(cell): owner for cell, owner in self._owners(self.position)},
            "turn": {} if self.position.turn is None else {"seat": self.seats[self.position.turn]},
            "last": {"move": self.last} if self.last else {},
        }

    def save_state(self) -> dict:
        return {
            "tick": self.tick,
            "pieces": {seat: list(cells_in(mask)) for seat, mask in zip(self.seats, self.position.pieces)},
            "turn": self.position.turn,
        }

    def _owners(self, position):
        for seat, mask in zip(self.seats, position.pieces):
            for cell in cells_in(mask):
                yield cell, seat


class WebPerson:
    """A person on the website. The move they last picked is played on their turn."""

    def __init__(self):
        self.move: Move | None = None

    def command(self, data) -> None:
        self.move = tuple(_checked(data))

    def act(self, position) -> Move | None:
        move, self.move = self.move, None
        return move

    def view(self) -> dict:
        return {"move": list(self.move)} if self.move else {}


class SpliceGame:
    info = INFO

    def maps(self) -> tuple[MapInfo, ...]:
        return tuple(MapInfo(layout.id, layout.name, SEATS[: layout.players], layout.description) for layout in MAPS.values())

    def bots(self) -> tuple[BotInfo, ...]:
        return tuple(
            BotInfo(id=name, name=NAMES.get(name) or name.removesuffix("_ai").title(), description=summary(ROOT / "ais" / f"{name}.py"))
            for name in _roster()
        )

    def load_bot(self, bot_id: str):
        if bot_id not in _roster():
            raise ValueError(f"Splice has no bot named {bot_id!r}")
        spec = importlib.util.spec_from_file_location(f"splice_ais_{bot_id}", ROOT / "ais" / f"{bot_id}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return bot_from_module(module)

    def new_match(self, map_id: str, seed: int) -> SpliceMatch:
        # Splice's rules have no chance in them, so the seed only matters to bots.
        return SpliceMatch(map_id)

    def action_to_json(self, action) -> list[int]:
        """A move is [from, to], two cell numbers."""

        return list(_checked(action))

    def action_from_json(self, data: list[int]) -> Move:
        return tuple(data)

    def person(self, seat: str) -> WebPerson:
        return WebPerson()

    def themes(self) -> tuple[dict, ...]:
        others = sorted(path for path in THEMES.glob("*.json") if path.name != "petri_dish.json")
        return (THEME, *(json.loads(path.read_text()) for path in others))


def _checked(move):
    """``move`` if it is two cell numbers, else ValueError or TypeError."""

    if not isinstance(move, (list, tuple)):
        raise TypeError(f"a move is [from, to], not {type(move).__name__}")
    if len(move) != 2 or not all(isinstance(cell, int) and not isinstance(cell, bool) for cell in move):
        raise ValueError(f"a move is [from, to], two cell numbers, not {move!r}")
    return move


def _roster() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in (ROOT / "ais").glob("*.py") if not path.stem.startswith("_")))


GAME = SpliceGame()
