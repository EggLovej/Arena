"""Splice's maps, drawn as text: ``.`` is an empty cell and ``1``, ``2`` and ``3`` are the first,
second and third player's starting pieces. Cells in a row are two characters apart and each row
is shifted one character from the next, so the text has the board's shape; a space where a cell
would be is a hole, which pieces can jump over but never land on.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from textwrap import dedent

from splice.rules import Board, Position


@dataclass(frozen=True)
class Layout:
    id: str
    name: str
    description: str
    drawing: str

    @cached_property
    def players(self) -> int:
        return max(int(mark) for mark in self.drawing if mark.isdigit())

    @cached_property
    def start(self) -> Position:
        """The first player to move, on the board with every starting piece."""

        cells, pieces = [], [0] * self.players
        for row, line in enumerate(dedent(self.drawing).strip("\n").splitlines()):
            for column, mark in enumerate(line):
                if mark == " ":
                    continue
                if mark.isdigit():
                    pieces[int(mark) - 1] |= 1 << len(cells)
                elif mark != ".":
                    raise ValueError(f"{self.id}: {mark!r} is not a cell")
                cells.append((column, row))
        return Position(Board(cells), tuple(pieces), 0)


MAPS = {
    layout.id: layout
    for layout in (
        Layout(
            "honeycomb",
            "Honeycomb",
            "The open hexagon, with three holes across its middle. Three pieces each, on alternate corners.",
            """
                1 . . . 2
               . . . . . .
              . . . . . . .
             . . . . . . . .
            2 . .       . . 1
             . . . . . . . .
              . . . . . . .
               . . . . . .
                1 . . . 2
            """,
        ),
        Layout(
            "atoll",
            "Atoll",
            "A ring of cells round a lagoon: the fight goes round it, or jumps across its narrow shore.",
            """
                1 . . . 2
               . . . . . .
              . . . . . . .
             . . .     . . .
            2 . .       . . 1
             . . .     . . .
              . . . . . . .
               . . . . . .
                1 . . . 2
            """,
        ),
        Layout(
            "rift",
            "Rift",
            "Two halves split by a rift that only a jump crosses, but for a cell at each end. Each player starts on both sides.",
            """
                1 . . . 2
               . . . . . .
              . . . . . . .
             . . . . . . . .
            .               .
             . . . . . . . .
              . . . . . . .
               . . . . . .
                2 . . . 1
            """,
        ),
        Layout(
            "trefoil",
            "Trefoil",
            "Three players on the hexagon, two opposite corners each, round three holes in the middle.",
            """
                1 . . . 2
               . . . . . .
              . . . . . . .
             . . . .   . . .
            3 . .   . . . . 3
             . . . .   . . .
              . . . . . . .
               . . . . . .
                2 . . . 1
            """,
        ),
    )
}
