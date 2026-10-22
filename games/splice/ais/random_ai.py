"""Makes any legal move, each as likely as the others."""

import random


class RandomBot:
    def reset(self, context):
        self.random = random.Random(context.seed)

    def act(self, position):
        return self.random.choice(position.moves())


def create_bot():
    return RandomBot()
