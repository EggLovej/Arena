"""Analytic baseline that uses direct shots and top/bottom boundary bank shots."""

from ricochet import BaseController, ControlAction, GameObservation, MatchContext, Vec2
from ricochet.bot import line_of_sight, threats


class BankShotController(BaseController):
    def reset(self, context: MatchContext) -> None:
        super().reset(context)
        self.strafe_sign = 1 if self.player_id == 0 else -1

    def act(self, observation: GameObservation) -> ControlAction:
        me, opponent = observation.me, observation.opponent
        direct = opponent.position - me.position
        direct_blocked = not line_of_sight(observation, me.position, opponent.position)
        use_bank = direct_blocked or (observation.tick // 75) % 3 == 2
        if use_bank:
            virtual_targets = (
                Vec2(opponent.position.x, -opponent.position.y),
                Vec2(opponent.position.x, 2 * observation.height - opponent.position.y),
            )
            reflected_target = min(
                virtual_targets,
                key=lambda target: (target - me.position).length(),
            )
            aim = reflected_target - me.position
        else:
            aim = direct

        danger = None
        for projectile in threats(observation):
            offset = projectile.position - me.position
            if projectile.velocity.dot(offset) < 0 and offset.length() < 115:
                danger = projectile
                break
        shield = danger is not None and me.energy > 20
        direction = direct.normalized()
        strafe = Vec2(-direction.y, direction.x) * self.strafe_sign
        edge_push = Vec2(0, 0)
        if me.position.y < 90:
            edge_push = Vec2(0, 0.8)
        elif me.position.y > observation.height - 90:
            edge_push = Vec2(0, -0.8)
        movement = (strafe + edge_push).limited()
        return ControlAction(
            move=movement,
            aim=aim,
            fire=me.energy >= 38 and not shield,
            shield=shield,
        )


def create_controller() -> BankShotController:
    return BankShotController()
