"""Direct-fire baseline with simple projectile evasion and energy management."""

from ricochet import BaseController, ControlAction, GameObservation, MatchContext, Vec2
from ricochet.bot import threats


class DirectController(BaseController):
    def reset(self, context: MatchContext) -> None:
        super().reset(context)
        self.strafe_sign = 1 if self.player_id == 0 else -1

    def act(self, observation: GameObservation) -> ControlAction:
        me = observation.me
        target = observation.opponent.position - me.position
        distance = target.length()
        nearest = min(
            threats(observation),
            key=lambda projectile: (projectile.position - me.position).length(),
            default=None,
        )
        shield = False
        dodge = Vec2(0, 0)
        if nearest is not None:
            offset = nearest.position - me.position
            approach = nearest.velocity.dot(offset)
            if offset.length() < 100 and approach < 0:
                shield = me.energy > 18
                dodge = Vec2(-nearest.velocity.y, nearest.velocity.x).normalized()
        if dodge.length() > 0:
            movement = dodge * self.strafe_sign
        else:
            toward = target.normalized()
            strafe = Vec2(-toward.y, toward.x) * self.strafe_sign
            radial = toward if distance > 430 else toward * -0.35
            movement = (strafe * 0.8 + radial).limited()
        return ControlAction(
            move=movement,
            aim=target,
            fire=me.energy >= 38 and not shield,
            shield=shield,
        )


def create_controller() -> DirectController:
    return DirectController()
