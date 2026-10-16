"""Shared visual theme for the atlas-style game interface."""

from pathlib import Path

import pygame


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

INK = (42, 40, 37)
PAPER = (239, 231, 214)
PAPER_LIGHT = (249, 243, 230)
PAPER_DARK = (218, 205, 181)
MUTED = (105, 96, 83)
DISABLED = (157, 148, 133)
RED = (187, 57, 51)
RED_DARK = (132, 39, 36)
BLUE = (44, 91, 166)
BLUE_DARK = (31, 61, 111)
GREEN = (54, 132, 91)
GREEN_DARK = (35, 91, 62)
MUSTARD = (211, 164, 47)
WHITE = (255, 250, 240)


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
