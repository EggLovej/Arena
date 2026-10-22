"""Splice: a turn-based game of territory on a board of hexagons.

``rules`` has the board, positions and moves; ``maps`` the boards drawn as text.
A bot is given the ``Position`` when it is its turn and answers with a ``Move``.
"""

from splice.maps import MAPS, Layout
from splice.rules import Board, Move, Position, cells_in

__all__ = ["MAPS", "Board", "Layout", "Move", "Position", "cells_in"]
