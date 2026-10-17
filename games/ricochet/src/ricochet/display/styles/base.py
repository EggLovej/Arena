"""What every theme shares: HUD and footer layout, the tank drawing pipeline with its
shield/hit/analysis overlays, and small cached drawing helpers.

A theme's style subclasses ``BaseStyle`` and decides how the arena and the machines look by
implementing ``floor`` and ``tank_body``. Colors that the HUD, menu and effects also use come
from the theme's palette; colors that only describe one look are named on its style class.
"""

import math
from abc import ABC, abstractmethod
from functools import lru_cache

import pygame
import pygame.gfxdraw

from ricochet.game.config import RulesConfig
from ricochet.game.types import EventKind, RenderSnapshot, TankView, Vec2
from ricochet.themes import Color, Theme


@lru_cache(maxsize=64)
def _halo(radius, color):
    extent = radius * 2
    halo = pygame.Surface((extent * 2 + 1, extent * 2 + 1))
    for distance in range(extent, 0, -1):
        strength = 0.65 * (1 - distance / extent) ** 2
        pygame.draw.circle(
            halo, tuple(round(c * strength) for c in color), (extent, extent), distance
        )
    return halo


def glow_circle(surface, center, radius, color):
    """Cached, bounded additive halo; never blurs collision geometry."""
    halo = _halo(radius, color)
    surface.blit(
        halo, (center[0] - radius * 2, center[1] - radius * 2), special_flags=pygame.BLEND_RGB_ADD
    )


@lru_cache(maxsize=16)
def hud_font(size):
    return pygame.font.SysFont("menlo,dejavusansmono,consolas,monospace", size)


@lru_cache(maxsize=4)
def status_font(size):
    return pygame.font.Font(None, size)


@lru_cache(maxsize=256)
def tracked_text(text, font, color, spacing):
    glyphs = [font.render(char, True, color) for char in text]
    width = sum(glyph.get_width() for glyph in glyphs) + spacing * max(0, len(glyphs) - 1)
    result = pygame.Surface((max(1, width), font.get_height()), pygame.SRCALPHA)
    x = 0
    for glyph in glyphs:
        result.blit(glyph, (x, 0))
        x += glyph.get_width() + spacing
    return result


def chamfer(rect, cut):
    """Outline of ``rect`` with all four corners cut at 45 degrees."""
    return [
        (rect.left + cut, rect.top),
        (rect.right - cut, rect.top),
        (rect.right, rect.top + cut),
        (rect.right, rect.bottom - cut),
        (rect.right - cut, rect.bottom),
        (rect.left + cut, rect.bottom),
        (rect.left, rect.bottom - cut),
        (rect.left, rect.top + cut),
    ]


class TankSprite:
    """A supersampled square centred on one tank. Lengths are given in screen pixels."""

    SCALE = 3

    def __init__(self, radius: float) -> None:
        self.size = round(radius * 5)
        self.surface = pygame.Surface((self.size * self.SCALE,) * 2, pygame.SRCALPHA)
        self.center = self.size * self.SCALE // 2

    def px(self, length: float) -> int:
        return round(length * self.SCALE)

    def point(self, along: float, across: float, direction: Vec2, unit: float = 1.0):
        """Sprite coordinates of the point ``along``/``across`` ``direction`` from the centre."""
        scale = self.SCALE * unit
        return (
            self.center + scale * (direction.x * along - direction.y * across),
            self.center + scale * (direction.y * along + direction.x * across),
        )

    def polygon(self, color, points, direction, unit=1.0):
        corners = [self.point(along, across, direction, unit) for along, across in points]
        pygame.draw.polygon(self.surface, color, corners)

    def line(self, color, start, end, direction, width):
        first, second = self.point(*start, direction), self.point(*end, direction)
        pygame.draw.line(self.surface, color, first, second, self.px(width))

    def circle(self, color, radius, width=0, offset=(0, 0)):
        center = (self.center + self.px(offset[0]), self.center + self.px(offset[1]))
        pygame.draw.circle(self.surface, color, center, self.px(radius), self.px(width))

    def arc(self, color, radius, start, stop, width):
        extent = self.px(radius)
        rect = pygame.Rect(self.center - extent, self.center - extent, 2 * extent, 2 * extent)
        pygame.draw.arc(self.surface, color, rect, start, stop, self.px(width))

    def blit_centered(self, image):
        self.surface.blit(image, image.get_rect(center=(self.center, self.center)))

    def draw(self, surface, center):
        image = pygame.transform.smoothscale(self.surface, (self.size, self.size))
        surface.blit(image, (center[0] - self.size // 2, center[1] - self.size // 2))


class BaseStyle(ABC):
    """Theme-independent layout and overlays. Subclasses own the arena and machine look."""

    RECOIL = 3  # Pixels the turret kicks back on the frame a shot is fired.
    SHIELD_GAP = 12  # Shield ring distance beyond the tank radius.

    def __init__(self, theme: Theme):
        self.theme = theme

    # Look of the theme -------------------------------------------------------------------

    @abstractmethod
    def floor(self, snapshot: RenderSnapshot) -> pygame.Surface:
        """Arena-sized static background, including walls; baked once per map."""

    @abstractmethod
    def tank_body(
        self, sprite: TankSprite, tank: TankView, radius: float, color: Color, recoil: float
    ) -> None:
        """Draw the machine itself (hull and turret) into the supersampled sprite."""

    def tank_underlay(self, surface, center, radius, color, analysis):  # noqa: B027
        """Optional glow or shadow drawn on the arena underneath the tank sprite."""

    def muzzle_flash(self, surface, tank, position, radius, recoil):  # noqa: B027
        """Optional effect at the barrel when a shot is fired (skipped in analysis mode)."""

    def wreck(self, surface, tank, center, radius):
        pygame.draw.circle(surface, self.theme.inactive, center, round(radius), 2)

    def shield_color(self, tank: TankView) -> Color:
        return self.theme.shield

    def projectile(self, surface, center, color, analysis):
        pygame.draw.circle(surface, color, center, 5)

    def trail(self, surface, first, second, color, index, count):
        shade = tuple(round(c * (0.2 + 0.6 * (index + 1) / count)) for c in color)
        pygame.draw.aaline(surface, shade, (first.x, first.y), (second.x, second.y))

    def particle(self, surface, position, velocity, life, color):
        pygame.draw.circle(surface, color, (round(position.x), round(position.y)), 2)

    # Shared drawing ------------------------------------------------------------------------

    def tank(
        self, surface, tank: TankView, position: Vec2, rules: RulesConfig, *, analysis, flashes
    ):
        radius = rules.tank_radius
        center = (round(position.x), round(position.y))
        if tank.health == 0:
            self.wreck(surface, tank, center, radius)
            return
        color = self.theme.players[tank.player_id]
        recoil = self.RECOIL if (EventKind.FIRED, tank.player_id) in flashes else 0
        self.tank_underlay(surface, center, radius, color, analysis)
        sprite = TankSprite(radius)
        shield = self.shield_color(tank)
        shield_radius = radius + self.SHIELD_GAP
        if tank.shield_active:
            # The translucent field goes first: drawn last it would wash out the turret.
            sprite.circle((*shield, 20), shield_radius)
        self.tank_body(sprite, tank, radius, color, recoil)
        if tank.shield_active:
            sprite.circle(shield, shield_radius, width=1)
            if tank.shield_ticks_remaining and rules.shield_min_ticks:
                committed = tank.shield_ticks_remaining / rules.shield_min_ticks
                sprite.arc(self.theme.text, shield_radius + 2, 0, math.tau * committed, 1)
        if (EventKind.HIT, tank.player_id) in flashes:
            sprite.circle(self.theme.text, radius, width=1)
        sprite.draw(surface, center)
        if recoil and not analysis:
            self.muzzle_flash(surface, tank, position, radius, recoil)
        if analysis:
            pygame.draw.circle(surface, self.theme.text, center, round(radius), 1)

    def hud(self, surface, snapshot: RenderSnapshot, names: tuple[str, str]):
        obs, rules = snapshot.observation, snapshot.rules
        width = surface.get_width()
        label_font = hud_font(15)
        panel_width = int(width * 0.30)
        for tank, x in ((obs.tanks[0], 15), (obs.tanks[1], width - panel_width - 15)):
            color = self.theme.players[tank.player_id]
            panel = pygame.Rect(x, 10, panel_width, 58)
            self.hud_panel(surface, panel, color)
            label = tracked_text(names[tank.player_id].upper(), label_font, color, 2)
            max_width = panel_width - 108
            if label.get_width() > max_width:
                label = pygame.transform.smoothscale(label, (max_width, label.get_height()))
            surface.blit(label, (x + 18, 18))
            for i in range(rules.starting_health):
                center = (x + panel_width - 82 + i * 18, 26)
                if i < tank.health:
                    glow_circle(surface, center, 7, color)
                light_color = color if i < tank.health else self.theme.inactive
                pygame.gfxdraw.aacircle(surface, *center, 6, light_color)
                pygame.gfxdraw.filled_circle(surface, *center, 6, light_color)
            bar = pygame.Rect(x + 18, 43, panel_width - 36, 13)
            pygame.draw.rect(surface, color, bar, 1)
            fill = bar.inflate(-4, -4)
            pygame.draw.rect(surface, self.theme.background, fill)
            fill.width = round(fill.width * max(0, min(1, tank.energy / rules.max_energy)))
            if fill.width:
                pygame.draw.rect(surface, color, fill)
                pygame.draw.line(
                    surface, tuple(min(255, v + 45) for v in color), fill.topleft, fill.topright
                )
        title = tracked_text("RICOCHET", hud_font(15), self.theme.text, 5)
        surface.blit(title, title.get_rect(midtop=(width // 2, 7)))
        timer = tracked_text(
            f"{max(0, obs.max_seconds - obs.time_seconds):04.1f}", hud_font(30), self.theme.text, 2
        )
        surface.blit(timer, timer.get_rect(midtop=(width // 2, 29)))
        for direction in (-1, 1):
            x = width // 2 + direction * 85
            pygame.draw.line(surface, self.theme.muted, (x, 19), (x + direction * 26, 19))
            pygame.draw.line(surface, self.theme.accent, (x, 24), (x + direction * 16, 24))
            pygame.draw.line(surface, self.theme.muted, (x, 58), (x + direction * 14, 58))

    def hud_panel(self, surface, rect, color):
        points = [
            rect.topleft,
            (rect.right - 14, rect.top),
            (rect.right, rect.top + 14),
            rect.bottomright,
            (rect.left + 14, rect.bottom),
            (rect.left, rect.bottom - 14),
        ]
        pygame.draw.polygon(surface, tuple(int(c * 0.7) for c in color), points, 1)
        pygame.draw.line(surface, color, rect.topleft, (rect.left + 44, rect.top), 2)

    def footer(self, surface, text, y, *, status=False):
        """Controls hint centred between rules; ``status`` lines (replay, analysis) left-aligned."""
        if status:
            surface.blit(status_font(18).render(text, True, self.theme.muted), (16, y + 10))
            return
        hint = tracked_text(text.upper(), hud_font(9), self.theme.muted, 2)
        surface.blit(hint, hint.get_rect(center=(surface.get_width() // 2, y + 17)))
        margin = (surface.get_width() - hint.get_width()) // 2 - 18
        for start, end in ((30, margin), (surface.get_width() - margin, surface.get_width() - 30)):
            pygame.draw.line(surface, self.theme.muted, (start, y + 17), (end, y + 17))
