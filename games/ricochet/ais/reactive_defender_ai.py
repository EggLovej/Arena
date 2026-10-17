"""Stationary analytical defense baseline; deliberately does not attack."""

from ricochet import BaseController, ControlAction, GameObservation
from ricochet.bot import on_collision_course, threats


class ReactiveDefender(BaseController):
    def act(self, observation: GameObservation) -> ControlAction:
        reach = self.rules.tank_radius + self.rules.projectile_radius
        threat = any(
            on_collision_course(projectile, observation.me.position, reach, self.rules.dt)
            for projectile in threats(observation)
        )
        return ControlAction(shield=threat)


def create_controller() -> ReactiveDefender:
    return ReactiveDefender()
