"""Immutable presentation definitions; no renderer, engine or optional dependencies.

A theme names its Pygame style (``arena_style``) and supplies the semantic palette that the
shared HUD, menu and effects draw with. Colors specific to one look live on that style.
"""

from dataclasses import dataclass

Color = tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    id: str
    name: str
    description: str
    arena_style: str
    background: Color
    panel: Color
    text: Color
    muted: Color
    players: tuple[Color, Color]
    accent: Color
    energy: Color
    inactive: Color
    shield: Color


NEON_CIRCUIT = Theme(
    "neon-circuit",
    "Neon Circuit",
    "Luminous geometry / midnight court",
    "neon",
    (9, 8, 24),
    (12, 12, 30),
    (236, 234, 255),
    (148, 141, 185),
    ((31, 229, 255), (255, 103, 135)),
    (166, 87, 255),
    (255, 209, 123),
    (42, 36, 65),
    (104, 236, 255),
)
ROBOT_LEAGUE = Theme(
    "robot-league",
    "Robot League",
    "Concrete court / armored machines",
    "robot-league",
    (16, 20, 24),
    (12, 16, 20),
    (233, 239, 241),
    (151, 168, 177),
    ((27, 206, 235), (246, 104, 86)),
    (229, 175, 37),
    (246, 195, 57),
    (45, 54, 60),
    (106, 223, 244),
)
DEFAULT_THEME = NEON_CIRCUIT.id
_THEMES = {theme.id: theme for theme in (NEON_CIRCUIT, ROBOT_LEAGUE)}


def available_themes() -> tuple[str, ...]:
    return tuple(_THEMES)


def get_theme(theme_id: str = DEFAULT_THEME) -> Theme:
    try:
        return _THEMES[theme_id]
    except KeyError:
        raise ValueError(f"Unknown theme {theme_id!r}; choose from {', '.join(_THEMES)}") from None
