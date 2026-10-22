"""Monte Carlo tree search: it tries the most promising moves, plays each line on a couple of
moves greedily, and keeps exploring the lines that turn out best for whoever chose them."""

import math
import random

from splice.search import gain, lead, ordered

ITERATIONS = 400  # lines tried a move
WIDTH = 12  # the moves tried in a position: those that gain most there
ROLLOUT = 2  # moves played greedily from the end of a line before reading the lead
EXPLORE = 0.3  # how much the search strays from the best lines to try others
SPREAD = 4  # a lead this many pieces bigger takes a player's reward most of the way from 0.5 to 1


class Node:
    def __init__(self, position):
        self.position = position
        self.untried = ordered(position)[:WIDTH] if position.turn is not None else []
        self.children = []
        self.visits = 0
        self.reward = 0.0  # summed over visits, for the player who moved into this position


class MctsBot:
    def reset(self, context):
        self.random = random.Random(context.seed)

    def act(self, position):
        root = Node(position)
        moves = {}
        for _ in range(ITERATIONS):
            path = [root]
            while not path[-1].untried and path[-1].children:
                path.append(self.select(path[-1]))
            node = path[-1]
            if node.untried:
                move = node.untried.pop(0)
                child = Node(node.position.play(move))
                node.children.append(child)
                if node is root:
                    moves[child] = move
                path.append(child)
            rewards = self.rollout(path[-1].position)
            root.visits += 1
            for parent, child in zip(path, path[1:]):
                child.visits += 1
                child.reward += rewards[parent.position.turn]
        return moves[max(root.children, key=lambda child: child.visits)]

    def select(self, node):
        """The child with the best reward so far, plus a bonus for having been tried less (UCB1)."""

        bonus = EXPLORE * math.sqrt(math.log(node.visits))
        return max(node.children, key=lambda child: child.reward / child.visits + bonus / math.sqrt(child.visits))

    def rollout(self, position):
        """Each player's reward, between 0 and 1, after a few greedy moves from ``position``."""

        for _ in range(ROLLOUT):
            if position.turn is None:
                break
            moves = position.moves()
            gains = [gain(position, move) for move in moves]
            most = max(gains)
            position = position.play(self.random.choice([move for move, value in zip(moves, gains) if value == most]))
        return [1 / (1 + math.exp(-lead(position, player) / SPREAD)) for player in range(len(position.pieces))]


def create_bot():
    return MctsBot()
