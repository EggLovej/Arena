"""Takes the most pieces it can this move, without a thought for the reply.
The leaderboard's fixed point: it is rated 1000."""

import random

from splice.search import gain


class GreedyBot:
    def reset(self, context):
        self.random = random.Random(context.seed)

    def act(self, position):
        moves = position.moves()
        most = max(gain(position, move) for move in moves)
        return self.random.choice([move for move in moves if gain(position, move) == most])


def create_bot():
    return GreedyBot()
