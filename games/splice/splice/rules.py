"""Splice's rules: pieces on a board of hexagons that grow, jump and win over their neighbours.

A board's cells are numbered in reading order, and a set of cells is a bit mask
(cell ``i`` is ``1 << i``), so a position is one whole number per player and a
move changes a few of them. That keeps the rules quick enough for bots that look
many moves ahead, in a browser too.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property

Move = tuple[int, int]  # (from, to): to a cell next to ``from`` grows a new piece; to one two away, the piece jumps


def cells_in(mask: int) -> Iterator[int]:
    """The cells in a mask, lowest first."""

    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Steps between two cells given as (column, row), where a row's cells are two columns apart
    and each row is shifted one column from the next."""

    columns, rows = abs(a[0] - b[0]), abs(a[1] - b[1])
    return rows + max(0, (columns - rows) // 2)


class Board:
    """The cells of a map, and for each cell the cells one and two steps from it, as masks."""

    def __init__(self, cells: list[tuple[int, int]]):
        self.cells = cells
        self.all = (1 << len(cells)) - 1
        self.near = [sum(1 << j for j, other in enumerate(cells) if distance(cell, other) == 1) for cell in cells]
        self.far = [sum(1 << j for j, other in enumerate(cells) if distance(cell, other) == 2) for cell in cells]
        self.reach = [near | far for near, far in zip(self.near, self.far)]


@dataclass(frozen=True)
class Position:
    """Everyone's pieces, in turn order, and whose move it is: ``turn`` is None once the game is over."""

    board: Board
    pieces: tuple[int, ...]
    turn: int | None

    @cached_property
    def empty(self) -> int:
        taken = 0
        for mask in self.pieces:
            taken |= mask
        return self.board.all & ~taken

    def counts(self) -> tuple[int, ...]:
        return tuple(mask.bit_count() for mask in self.pieces)

    def moves(self) -> list[Move]:
        """The legal moves of the player whose turn it is: one grow into each cell they can grow
        into (from which piece makes no difference), then every jump."""

        board, mine, empty = self.board, self.pieces[self.turn], self.empty
        grown = 0
        for cell in cells_in(mine):
            grown |= board.near[cell]
        grows = [(next(cells_in(board.near[cell] & mine)), cell) for cell in cells_in(grown & empty)]
        return grows + [(cell, target) for cell in cells_in(mine) for target in cells_in(board.far[cell] & empty)]

    def refusal(self, move: Move) -> str | None:
        """Why the player to move can't make this move, or None if they can."""

        start, end = move
        cells = len(self.board.cells)
        if not (0 <= start < cells and 0 <= end < cells):
            return f"the board has cells 0 to {cells - 1}"
        if not self.pieces[self.turn] >> start & 1:
            return f"cell {start} is not yours"
        if not self.empty >> end & 1:
            return f"cell {end} is taken"
        if not self.board.reach[start] >> end & 1:
            return f"cell {end} is more than two steps from cell {start}"
        return None

    def play(self, move: Move) -> Position:
        """The position after the player to move makes a legal ``move``. Every other player's
        piece next to where it lands becomes theirs. Then the turn passes to the next player
        who can move; when only one player can, the empty cells are theirs and the game is over."""

        start, end = move
        board, me = self.board, self.turn
        pieces = list(self.pieces)
        mine = pieces[me] | 1 << end
        if board.far[start] >> end & 1:
            mine &= ~(1 << start)
        for player, theirs in enumerate(pieces):
            if player != me:
                taken = theirs & board.near[end]
                pieces[player] = theirs & ~taken
                mine |= taken
        pieces[me] = mine
        return after(board, pieces, me)


def can_move(board: Board, mask: int, empty: int) -> bool:
    """Whether the pieces in ``mask`` can reach an empty cell."""

    return any(board.reach[cell] & mask for cell in cells_in(empty))


def after(board: Board, pieces: list[int], moved: int) -> Position:
    """The position once ``moved`` has played: whose turn it is, or, when one player is left who
    can move, that player's pieces filling every empty cell and the game over."""

    taken = 0
    for mask in pieces:
        taken |= mask
    empty = board.all & ~taken
    count = len(pieces)
    able = [player for player in range(count) if pieces[player] and can_move(board, pieces[player], empty)]
    if len(able) > 1:
        turn = min(able, key=lambda player: (player - moved - 1) % count)
        return Position(board, tuple(pieces), turn)
    if able:
        pieces[able[0]] |= empty
    return Position(board, tuple(pieces), None)
