"""Shared visual theme for the atlas-style game interface.

The colours live in ``themes/paper_atlas.json``, which is also the website's
default board theme, so the game window and the site match.
"""

import json
from pathlib import Path

import pygame


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
THEME = json.loads((ROOT / "themes" / "paper_atlas.json").read_text())


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[index : index + 2], 16) for index in (1, 3, 5))


_COLORS = THEME["colors"]
_PLAYERS = THEME["players"]
INK = _rgb(_COLORS["ink"])
PAPER = _rgb(_COLORS["paper"])
PAPER_LIGHT = _rgb(_COLORS["paper_light"])
PAPER_DARK = _rgb(_COLORS["paper_dark"])
MUTED = _rgb(_COLORS["muted"])
DISABLED = _rgb(_COLORS["disabled"])
RED = _rgb(_PLAYERS["red"]["fill"])
BLUE = _rgb(_PLAYERS["blue"]["fill"])
GREEN = _rgb(_PLAYERS["green"]["fill"])
MUSTARD = _rgb(_COLORS["highlight"])
WHITE = _rgb(_COLORS["white"])


def font(size: int, bold: bool = False) -> pygame.font.Font:
    path = pygame.font.match_font("dejavusanscondensed", bold=bold)
    return pygame.font.Font(path, size)


def load_background(size: tuple[int, int]) -> pygame.Surface:
    image = pygame.image.load(ASSETS / "paper-atlas-background.png").convert()
    return pygame.transform.smoothscale(image, size)


def load_brush(filename: str, size: tuple[int, int]) -> pygame.Surface:
    image = pygame.image.load(ASSETS / filename).convert_alpha()
    visible_area = image.get_bounding_rect(min_alpha=10)
    cropped = image.subsurface(visible_area).copy()
    return pygame.transform.smoothscale(cropped, size)


def owner_color(owner: int) -> tuple[int, int, int]:
    if owner == -1:
        return RED
    if owner == 1:
        return BLUE
    if owner == 2:
        return GREEN
    return PAPER_DARK
