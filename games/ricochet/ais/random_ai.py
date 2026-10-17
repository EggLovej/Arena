"""Seeded random baseline."""

import math
import random

from ricochet import BaseController, ControlAction, GameObservation, MatchContext, Vec2


class RandomController(BaseController):
    def reset(self, context: MatchContext) -> None:
        super().reset(context)
        self.random = random.Random(context.seed * 1009 + context.player_id * 9176 + 23)
        self.move_angle = self.random.random() * math.tau

    def act(self, observation: GameObservation) -> ControlAction:
        if observation.tick % 30 == 0:
            self.move_angle += self.random.uniform(-1.5, 1.5)
        aim_angle = self.random.random() * math.tau
        return ControlAction(
            move=Vec2(math.cos(self.move_angle), math.sin(self.move_angle)),
            aim=Vec2(math.cos(aim_angle), math.sin(aim_angle)),
            fire=self.random.random() < 0.08,
            shield=observation.me.energy > 25 and self.random.random() < 0.015,
        )


def create_controller() -> RandomController:
    return RandomController()
