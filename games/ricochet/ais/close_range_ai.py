"""Evaluation-only pursuit bot: navigate around walls and attack at short range."""

from ricochet import BaseController, ControlAction, GameObservation, MatchContext, Vec2
from ricochet.bot import find_path, lead_target


class CloseRange(BaseController):
    def reset(self, context: MatchContext) -> None:
        super().reset(context)
        self.route = []

    def act(self, observation: GameObservation) -> ControlAction:
        me, enemy = observation.me, observation.opponent
        if observation.tick % 15 == 0 or not self.route:
            self.route = find_path(
                observation, me.position, enemy.position, self.rules.tank_radius + 2
            )
        while self.route and (self.route[0] - me.position).length() < 20:
            self.route.pop(0)
        target = self.route[0] if self.route else enemy.position
        distance = (enemy.position - me.position).length()
        move = (target - me.position).normalized() if distance > 100 else Vec2(0, 0)
        if me.energy < self.rules.fire_energy and distance < 200:
            move = Vec2(0, 0)
        aim = lead_target(me.position, enemy, self.rules.projectile_speed)
        return ControlAction(move=move, aim=aim, fire=distance < 220)


def create_controller() -> CloseRange:
    return CloseRange()
