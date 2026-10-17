from __future__ import annotations

from typing import Protocol, runtime_checkable

from ricochet.game.types import (
    ControlAction,
    GameObservation,
    MatchContext,
    Vec2,
    create_noop_action,
    validate_action,
)


@runtime_checkable
class Controller(Protocol):
    def reset(self, context: MatchContext) -> None: ...

    def act(self, observation: GameObservation) -> ControlAction: ...


__all__ = [
    "ControlAction",
    "Controller",
    "GameObservation",
    "MatchContext",
    "Vec2",
    "create_noop_action",
    "validate_action",
]
