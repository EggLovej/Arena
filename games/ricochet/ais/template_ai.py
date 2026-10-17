"""Smallest useful Ricochet controller; copy this file to start a new bot."""

from ricochet import BaseController, ControlAction, GameObservation, Vec2


class TemplateController(BaseController):
    def act(self, observation: GameObservation) -> ControlAction:
        target = observation.opponent.position - observation.me.position
        return ControlAction(
            move=Vec2(0.0, 0.0),
            aim=target,
            fire=observation.me.energy >= 40,
            shield=False,
        )


def create_controller() -> TemplateController:
    return TemplateController()
