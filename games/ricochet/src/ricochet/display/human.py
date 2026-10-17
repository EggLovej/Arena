from __future__ import annotations

import pygame

from ricochet.bot import BaseController
from ricochet.display import controls
from ricochet.game.types import ControlAction, GameObservation, Vec2


class HumanController(BaseController):
    """Keyboard and mouse controller that emits the public AI action type."""

    def __init__(self, viewport=None):
        self.viewport = viewport

    def act(self, observation: GameObservation) -> ControlAction:
        keys = pygame.key.get_pressed()
        move = Vec2(
            float(keys[controls.MOVE_RIGHT]) - float(keys[controls.MOVE_LEFT]),
            float(keys[controls.MOVE_DOWN]) - float(keys[controls.MOVE_UP]),
        ).limited()
        mouse = pygame.mouse.get_pos()
        world = self.viewport.screen_to_world(mouse) if self.viewport else Vec2(*mouse)
        aim = world - observation.me.position
        buttons = pygame.mouse.get_pressed(3)
        return ControlAction(
            move=move,
            aim=aim,
            fire=bool(buttons[controls.FIRE_BUTTON]),
            shield=bool(buttons[controls.SHIELD_BUTTON] or keys[controls.SHIELD_KEY]),
        )
