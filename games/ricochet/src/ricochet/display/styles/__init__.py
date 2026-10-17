"""Pygame drawing styles. A theme's ``arena_style`` picks one; theme data stays Pygame-free."""

from ricochet.display.styles.base import BaseStyle
from ricochet.display.styles.neon import NeonCircuitStyle
from ricochet.display.styles.robot_league import RobotLeagueStyle
from ricochet.themes import Theme

STYLES: dict[str, type[BaseStyle]] = {
    "neon": NeonCircuitStyle,
    "robot-league": RobotLeagueStyle,
}


def create_style(theme: Theme) -> BaseStyle:
    try:
        style = STYLES[theme.arena_style]
    except KeyError:
        raise ValueError(f"Unsupported Pygame arena style: {theme.arena_style!r}") from None
    return style(theme)


__all__ = ["STYLES", "BaseStyle", "create_style"]
