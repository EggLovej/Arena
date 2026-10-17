"""Neon Circuit: midnight grid court, violet light-edged walls and segmented glowing robots."""

import math

import pygame

from ricochet.display.styles.base import BaseStyle, TankSprite, chamfer, glow_circle


def add_bloom(surface, light):
    """Multiscale soft light; used only while baking static map surfaces."""
    for factor in (4, 10, 24):
        reduced = pygame.transform.smoothscale(
            light, (max(1, light.get_width() // factor), max(1, light.get_height() // factor))
        )
        blur = pygame.transform.smoothscale(reduced, light.get_size())
        surface.blit(blur, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface.blit(light, (0, 0), special_flags=pygame.BLEND_RGB_ADD)


class NeonCircuitStyle(BaseStyle):
    # Court: a vertical gradient that lifts toward the middle, under a fine grid.
    FIELD_EDGE = (10, 9, 25)
    FIELD_MIDDLE = (13, 11, 34)
    GRID = (25, 21, 48)
    GRID_NODE = (57, 41, 88)
    # Light layer (bloomed): wall edges, infield line, corner brackets and edge ticks.
    WALL_FILL = (23, 14, 45)
    WALL_GLOW = (151, 62, 255)
    WALL_INNER_GLOW = (80, 28, 145)
    WALL_CORNER = (220, 183, 255)
    INFIELD = (156, 95, 236)
    BRACKET = (224, 185, 255)
    TICK = (210, 161, 255)
    # Sharp cores drawn above the bloom so obstacle footprints stay exact.
    WALL_CORE = (220, 173, 255)
    BORDER_CORE = (217, 167, 255)
    # Robots
    HULL_SHADOW = (4, 8, 18)
    HULL = (14, 22, 36)
    CORE = (13, 27, 42)
    HUB = (10, 29, 43)

    def floor(self, snapshot):
        obs = snapshot.observation
        width, height = int(obs.width), int(obs.height)
        surface = pygame.Surface((width, height))
        for y in range(height):
            lift = 1 - abs(y - height / 2) / (height / 2)
            shade = tuple(
                edge + int((middle - edge) * lift)
                for edge, middle in zip(self.FIELD_EDGE, self.FIELD_MIDDLE, strict=True)
            )
            pygame.draw.line(surface, shade, (0, y), (width, y))
        for x in range(0, width, 44):
            pygame.draw.line(surface, self.GRID, (x, 0), (x, height))
        for y in range(0, height, 44):
            pygame.draw.line(surface, self.GRID, (0, y), (width, y))
            for x in range(0, width, 44):
                surface.set_at((x, y), self.GRID_NODE)
        light = pygame.Surface((width, height))
        for wall in obs.walls:
            rect = pygame.Rect(wall.x, wall.y, wall.width, wall.height)
            pygame.draw.rect(surface, self.WALL_FILL, rect)
            pygame.draw.rect(light, self.WALL_GLOW, rect, 2)
            inner = rect.inflate(-12, -12)
            pygame.draw.rect(light, self.WALL_INNER_GLOW, inner, 1)
            for outer, inside in zip(
                (rect.topleft, rect.topright, rect.bottomleft, rect.bottomright),
                (inner.topleft, inner.topright, inner.bottomleft, inner.bottomright),
                strict=True,
            ):
                pygame.draw.line(light, self.WALL_CORNER, outer, inside, 1)
        border = surface.get_rect().inflate(-4, -4)
        pygame.draw.rect(light, self.WALL_GLOW, border, 2)
        # Chamfered infield line and doubled corner brackets.
        left, top, right, bottom = 11, 11, width - 12, height - 12
        infield = pygame.Rect(left, top, right - left, bottom - top)
        pygame.draw.lines(light, self.INFIELD, True, chamfer(infield, 9), 1)
        for x, sx in ((left, 1), (right, -1)):
            for y, sy in ((top, 1), (bottom, -1)):
                pygame.draw.lines(
                    light,
                    self.BRACKET,
                    False,
                    [
                        (x - 4 * sx, y + 20 * sy),
                        (x - 4 * sx, y + 5 * sy),
                        (x + 5 * sx, y - 4 * sy),
                        (x + 20 * sx, y - 4 * sy),
                        (x + 24 * sx, y - 9 * sy),
                    ],
                    1,
                )
        for x in (width // 5, width // 2, width * 4 // 5):
            for y in (5, height - 10):
                for dx in (0, 5):
                    pygame.draw.line(light, self.TICK, (x + dx, y), (x + dx, y + 4))
        for y in (height // 4, height * 3 // 4):
            for x in (5, width - 10):
                for dy in (0, 5):
                    pygame.draw.line(light, self.TICK, (x, y + dy), (x + 4, y + dy))
        add_bloom(surface, light)
        for wall in obs.walls:
            rect = pygame.Rect(wall.x, wall.y, wall.width, wall.height)
            pygame.draw.rect(surface, self.WALL_CORE, rect, 1)
        pygame.draw.rect(surface, self.BORDER_CORE, border, 1)
        return surface

    def shield_color(self, tank):
        return self.theme.players[tank.player_id]

    def tank_underlay(self, surface, center, radius, color, analysis):
        if not analysis:
            glow_circle(surface, center, round(radius), color)

    def tank_body(self, sprite: TankSprite, tank, radius, color, recoil):
        sprite.circle(self.HULL_SHADOW, radius, offset=(2, 3))
        sprite.circle(self.HULL, radius)
        # Four armor segments rotate with the turret.
        heading = math.atan2(-tank.aim.y, tank.aim.x)
        for segment in range(4):
            angle = heading + segment * math.pi / 2
            sprite.arc(tuple(int(v * 0.42) for v in color), radius, angle + 0.10, angle + 1.40, 4)
            sprite.arc(color, radius, angle + 0.16, angle + 1.14, 2)
        sprite.circle(tuple(int(v * 0.32) for v in color), radius - 5)
        sprite.circle(color, radius - 7)
        sprite.circle(self.CORE, radius - 10)
        # Broad solid turret, with a separate muzzle cap rather than a thin aim line.
        reach = radius + 7 - recoil
        sprite.polygon(color, [(2, -4), (reach, -4), (reach, 4), (2, 4)], tank.aim)
        bright = tuple(min(255, v + 40) for v in color)
        sprite.line(bright, (reach - 3, -3), (reach - 3, 3), tank.aim, 2)
        sprite.circle(self.HUB, 7)
        sprite.circle(color, 5)

    def projectile(self, surface, center, color, analysis):
        if not analysis:
            glow_circle(surface, center, 6, color)
        pygame.draw.circle(surface, color, center, 5)

    def trail(self, surface, first, second, color, index, count):
        end = first + (second - first) * 0.65
        brightness = 0.20 + 0.65 * (index + 1) / count
        faded = tuple(round(c * brightness) for c in color)
        pygame.draw.aaline(surface, faded, (first.x, first.y), (end.x, end.y))

    def particle(self, surface, position, velocity, life, color):
        end = position - velocity * min(0.045, life * 0.2)
        faded = tuple(round(c * min(1, life * 5)) for c in color)
        pygame.draw.aaline(surface, faded, (end.x, end.y), (position.x, position.y))
        if life > 0.22:
            glow_circle(surface, (round(position.x), round(position.y)), 3, color)
