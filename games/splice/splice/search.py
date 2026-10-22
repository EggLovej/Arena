"""Looking ahead, for the bots: how good a position is for a player, and the move that is best
after a few moves by everyone, found by alpha-beta search.

With three players the search is paranoid: it assumes both others play against the player
searching, which keeps alpha-beta's pruning and makes for a cautious player.
"""

from __future__ import annotations

import random

from splice.rules import Move, Position

WIN = 1000  # added to the lead in a finished game, so a sure win beats any lead before the end


def gain(position: Position, move: Move) -> int:
    """How many pieces a move gains its player: the pieces it wins over, and one more for a grow."""

    start, end = move
    board = position.board
    others = board.all & ~position.empty & ~position.pieces[position.turn]
    return (board.near[end] & others).bit_count() + (board.near[start] >> end & 1)


def lead(position: Position, player: int) -> int:
    """The player's pieces less those of the strongest other player, with ``WIN`` added or taken
    away once the game is over."""

    counts = position.counts()
    ahead = counts[player] - max(count for other, count in enumerate(counts) if other != player)
    if position.turn is None and ahead:
        return ahead + (WIN if ahead > 0 else -WIN)
    return ahead


def ordered(position: Position) -> list[Move]:
    """The moves worth searching for the player to move, those that gain most first, since
    alpha-beta prunes more when the best move comes early. Jumps that win nothing are left
    out unless there is nothing else: they gain nothing and leave a hole behind."""

    gains = [(gain(position, move), move) for move in position.moves()]
    worth = [pair for pair in gains if pair[0]] or gains
    worth.sort(key=lambda pair: -pair[0])
    return [move for _, move in worth]


def best_moves(position: Position, depth: int) -> list[Move]:
    """The moves for the player to move that lead to the best lead after ``depth`` moves by
    everyone (``depth`` 1 looks at this move alone), tied best moves in the order they were tried."""

    player = position.turn
    best, chosen = -WIN * 2, []
    for move in ordered(position):
        # A move that only ties the best so far still has to prove it ties, so the window starts just below it.
        value = _value(position.play(move), player, depth - 1, best - 1, WIN * 2)
        if value > best:
            best, chosen = value, [move]
        elif value == best:
            chosen.append(move)
    return chosen


def _value(position: Position, player: int, depth: int, alpha: int, beta: int) -> int:
    """The player's lead with best play for ``depth`` more moves, from their point of view: they
    take the most for themselves and everyone else the least for them."""

    if depth == 0 or position.turn is None:
        return lead(position, player)
    mine = position.turn == player
    for move in ordered(position):
        value = _value(position.play(move), player, depth - 1, alpha, beta)
        if mine:
            alpha = max(alpha, value)
        else:
            beta = min(beta, value)
        if alpha >= beta:
            break
    return alpha if mine else beta


class Searcher:
    """A bot that plays one of the best moves ``depth`` moves deep, chosen by chance among equals."""

    def __init__(self, depth: int):
        self.depth = depth

    def reset(self, context) -> None:
        self.random = random.Random(context.seed)

    def act(self, position: Position) -> Move:
        return self.random.choice(best_moves(position, self.depth))
