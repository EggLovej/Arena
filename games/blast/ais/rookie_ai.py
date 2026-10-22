"""Wanders about at random, out of the way of blasts, and now and then drops a bomb where it can
get away from it. The beginner every other bot should beat.
"""

from blast import MOVES, Bot

BOMB_CHANCE = 0.15  # each time it can step
KEEP_GOING = 0.7  # the chance it carries on the way it was going, when it can


class Rookie(Bot):
    def reset(self, context):
        super().reset(context)
        self.going = None

    def choose(self, observation, plan):
        if self.chance.random() < BOMB_CHANCE and plan.can_bomb():
            return "bomb"
        x, y = observation.me.tile
        ways = [way for way, (dx, dy) in MOVES.items() if plan.safe((x + dx, y + dy)) and plan.routes[(x + dx, y + dy)].first == way]
        if not ways:
            return None
        if self.going not in ways or self.chance.random() > KEEP_GOING:
            self.going = self.chance.choice(ways)
        dx, dy = MOVES[self.going]
        return plan.action(plan.routes[(x + dx, y + dy)])


def create_bot():
    return Rookie()
