"""Robot League: worn concrete court, steel rails, safety paint and tracked machines.

All coordinates are presentation only. Hazard paint uses the theme's accent color.
"""

import math
import random
from functools import lru_cache

import pygame

from ricochet.display.styles.base import BaseStyle, TankSprite, chamfer, glow_circle
from ricochet.resources import package_data

DESIGN_RADIUS = 18  # Machine art is drawn for this tank radius and scales with the rules.


@lru_cache(maxsize=3)
def material(name):
    with package_data("themes", "robot_league", name + ".png").open("rb") as stream:
        return pygame.image.load(stream, name + ".png")


@lru_cache(maxsize=16)
def chassis(color, pixels):
    image = pygame.transform.smoothscale(material("chassis"), (pixels, pixels))
    # Color only the armor; the black rubber tracks retain neutral highlights.
    tint = pygame.Surface(image.get_size(), pygame.SRCALPHA)
    tint.fill((255, 255, 255, 255))
    pygame.draw.rect(tint, (*color, 255), (pixels * 0.24, 0, pixels * 0.52, pixels))
    image.blit(tint, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return image


@lru_cache(maxsize=1)
def light_pool(color):
    # Sample a smooth radial falloff at low resolution, then interpolate it once.
    small = pygame.Surface((40, 40))
    for y in range(40):
        for x in range(40):
            distance = math.hypot((x - 19.5) / 19.5, (y - 19.5) / 19.5)
            power = max(0, 1 - distance) ** 2
            small.set_at((x, y), tuple(round(c * power) for c in color))
    return pygame.transform.smoothscale(small, (140, 140))


class RobotLeagueStyle(BaseStyle):
    # Concrete court
    CONCRETE_WASH = (104, 110, 115, 40)
    JOINT_SHADOW = (98, 102, 103)
    JOINT_HIGHLIGHT = (169, 171, 170)
    JOINT_SIDE = (107, 111, 112)
    JOINT_PIN = (100, 104, 105)
    WEAR = (47, 48, 47)
    TREAD_SCUFF = (42, 45, 44, 19)
    EDGE_SHADE = (13, 19, 27)
    # Perimeter rails, armor and paint
    FRAME = (30, 37, 43)
    FRAME_GROOVE = (9, 14, 18)
    FRAME_EDGE = (138, 151, 157)
    RAIL_EDGE = (112, 124, 132)
    SIDE_RAIL = (78, 88, 94)
    CORNER_ARMOR = (62, 72, 79)
    CORNER_EDGE = (167, 177, 180)
    HAZARD_STRIPE = (57, 56, 45)
    # Lamps
    LAMP_HOUSING = (11, 15, 19)
    LAMP_RIM = (158, 143, 111)
    LAMP_LENS = (255, 210, 133)
    LAMP_CORE = (255, 250, 230)
    LAMP_GLOW = (255, 206, 122)
    LAMP_SPILL = (95, 72, 35)
    # Bolts
    BOLT_SHADOW = (18, 23, 28)
    BOLT_RIM = (155, 166, 171)
    BOLT_FACE = (53, 62, 69)
    BOLT_GLINT = (206, 214, 216)
    # Steel walls and soft shadows
    SHADOW = (7, 12, 17)
    WALL_BODY = (25, 30, 34)
    WALL_EDGE = (165, 177, 182)
    WALL_BEVEL = (99, 111, 117)
    # Machines
    BARREL_SHADOW = (18, 27, 32)
    BARREL = (139, 155, 164)
    HUB_SHADOW = (19, 29, 35)
    HUB_RIM = (164, 179, 184)
    HUB_CORE = (39, 58, 68)
    HUB_CAP = (120, 159, 175)
    WRECK = (75, 77, 76)

    def floor(self, snapshot):
        obs = snapshot.observation
        size = (int(obs.width), int(obs.height))
        concrete = pygame.transform.smoothscale(material("concrete"), size)
        # Preserve material detail without letting high-frequency grit dominate the court.
        soft = pygame.transform.smoothscale(concrete, (size[0] // 3, size[1] // 3))
        floor = pygame.transform.smoothscale(soft, size)
        concrete.set_alpha(65)
        floor.blit(concrete, (0, 0))
        wash = pygame.Surface(size, pygame.SRCALPHA)
        wash.fill(self.CONCRETE_WASH)
        floor.blit(wash, (0, 0))
        # Local seeded wear never consumes the engine's RNG.
        rng = random.Random(813)
        for y in range(0, size[1], 128):
            for x in range(0, size[0], 160):
                pygame.draw.line(floor, self.JOINT_SHADOW, (x, y), (x + 160, y))
                pygame.draw.line(floor, self.JOINT_HIGHLIGHT, (x, y + 1), (x + 160, y + 1))
                pygame.draw.line(floor, self.JOINT_SIDE, (x, y), (x, y + 128))
                for dx, dy in ((4, 4), (155, 4), (4, 123), (155, 123)):
                    pygame.draw.circle(floor, self.JOINT_PIN, (x + dx, y + dy), 1)
        wear = pygame.Surface(size, pygame.SRCALPHA)
        for _ in range(90):
            x, y = rng.randrange(20, size[0] - 20), rng.randrange(20, size[1] - 20)
            pygame.draw.aaline(
                wear,
                (*self.WEAR, rng.randrange(18, 45)),
                (x, y),
                (x + rng.randrange(-12, 13), y + rng.randrange(3, 17)),
            )
        # A few faded tread scuffs rather than a noisy repeated floor pattern.
        for start_x, start_y, angle in (
            (80, size[1] * 0.65, -0.5),
            (size[0] * 0.7, 140, -0.55),
            (size[0] * 0.58, size[1] * 0.8, 0.3),
        ):
            for step in range(26):
                for side in (-1, 1):
                    x = start_x + step * 4 * math.cos(angle) - side * 8 * math.sin(angle)
                    y = start_y + step * 4 * math.sin(angle) + side * 8 * math.cos(angle)
                    pygame.draw.line(
                        wear,
                        self.TREAD_SCUFF,
                        (x, y),
                        (x - 4 * math.sin(angle), y + 4 * math.cos(angle)),
                        2,
                    )
        floor.blit(wear, (0, 0))
        # Perimeter frame sits inside the non-traversable center margin (tank radius).
        pygame.draw.rect(floor, self.FRAME, floor.get_rect(), 15)
        pygame.draw.rect(floor, self.FRAME_GROOVE, floor.get_rect().inflate(-28, -28), 3)
        pygame.draw.rect(floor, self.FRAME_EDGE, floor.get_rect().inflate(-2, -2), 1)
        for x in range(20, size[0] - 20, 80):
            for y in (2, size[1] - 13):
                rect = pygame.Rect(x, y, 73, 10)
                floor.blit(pygame.transform.smoothscale(material("steel"), rect.size), rect)
                pygame.draw.rect(floor, self.RAIL_EDGE, rect, 1)
                self._bolt(floor, (x + 5, y + 5), 2)
                self._bolt(floor, (x + 68, y + 5), 2)
        for y in range(23, size[1] - 24, 60):
            for x in (3, size[0] - 12):
                pygame.draw.rect(floor, self.SIDE_RAIL, (x, y, 8, 54), 1)
                self._bolt(floor, (x + 4, y + 5), 2)
        # Broad safety paint frames the floor, with deliberate breaks at the light wells.
        for x in (37, size[0] - 167):
            for y in (25, size[1] - 39):
                self._hazard(floor, pygame.Rect(x, y, 130, 14))
        for y in (61, size[1] - 196):
            for x in (25, size[0] - 39):
                self._hazard(floor, pygame.Rect(x, y, 14, 135))
        for y in (size[1] // 2 - 66,):
            for x in (25, size[0] - 35):
                self._hazard(floor, pygame.Rect(x, y, 10, 132))
        # Corner armor and bright coupler plates give the surround physical weight.
        for x, sx in ((0, 1), (size[0], -1)):
            for y, sy in ((0, 1), (size[1], -1)):
                points = [(x, y), (x + sx * 32, y), (x + sx * 13, y + sy * 26), (x, y + sy * 32)]
                pygame.draw.polygon(floor, self.CORNER_ARMOR, points)
                pygame.draw.lines(floor, self.CORNER_EDGE, False, points[1:], 1)
                pygame.draw.line(
                    floor,
                    self.theme.accent,
                    (x + sx * 9, y + sy * 25),
                    (x + sx * 25, y + sy * 9),
                    5,
                )
        for x in (size[0] // 3, size[0] * 2 // 3):
            for y in (2, size[1] - 13):
                pygame.draw.rect(floor, self.theme.accent, (x - 12, y, 24, 11))
                self._bolt(floor, (x - 8, y + 5), 2)
                self._bolt(floor, (x + 8, y + 5), 2)
        # Low-frequency edge shading establishes a well-lit court recessed inside steel rails.
        shading = pygame.Surface((64, 48), pygame.SRCALPHA)
        for y in range(48):
            for x in range(64):
                edge = max(abs(x / 63 * 2 - 1), abs(y / 47 * 2 - 1))
                shading.set_at((x, y), (*self.EDGE_SHADE, round(65 * edge**5)))
        floor.blit(pygame.transform.smoothscale(shading, size), (0, 0))
        # Additive light pools are baked once and can actually brighten the concrete.
        for x in (size[0] // 8, size[0] * 3 // 8, size[0] * 5 // 8, size[0] * 7 // 8):
            self._lamp(floor, (x, 13), (0, 1))
            self._lamp(floor, (x, size[1] - 14), (0, -1))
        for y in (size[1] // 4, size[1] * 3 // 4):
            self._lamp(floor, (13, y), (1, 0))
            self._lamp(floor, (size[0] - 14, y), (-1, 0))
        for wall in obs.walls:
            self._wall(floor, pygame.Rect(wall.x, wall.y, wall.width, wall.height))
        return floor

    def _bolt(self, surface, point, radius=3):
        x, y = point
        pygame.draw.circle(surface, self.BOLT_SHADOW, (x, y + 1), radius + 1)
        pygame.draw.circle(surface, self.BOLT_RIM, point, radius)
        pygame.draw.circle(surface, self.BOLT_FACE, point, max(1, radius - 1))
        pygame.draw.line(surface, self.BOLT_GLINT, (x - 1, y - 1), (x + 1, y - 1))

    def _hazard(self, surface, rect):
        previous = surface.get_clip()
        surface.set_clip(rect.clip(previous))
        pygame.draw.rect(surface, self.theme.accent, rect)
        for x in range(rect.left - rect.height, rect.right + 16, 16):
            pygame.draw.polygon(
                surface,
                self.HAZARD_STRIPE,
                [
                    (x, rect.bottom),
                    (x + 7, rect.bottom),
                    (x + rect.height + 7, rect.top),
                    (x + rect.height, rect.top),
                ],
            )
        surface.set_clip(previous)

    def _lamp(self, surface, center, direction):
        x, y = center
        dx, dy = direction
        pool = light_pool(self.LAMP_SPILL)
        surface.blit(
            pool,
            (x + dx * 24 - pool.get_width() // 2, y + dy * 24 - pool.get_height() // 2),
            special_flags=pygame.BLEND_RGB_ADD,
        )
        rect = pygame.Rect(0, 0, 29 if dy else 10, 10 if dy else 29)
        rect.center = center
        pygame.draw.rect(surface, self.LAMP_HOUSING, rect.inflate(5, 4), border_radius=2)
        pygame.draw.rect(surface, self.LAMP_RIM, rect, 1, border_radius=2)
        pygame.draw.rect(surface, self.LAMP_LENS, rect.inflate(-4, -4), border_radius=1)
        pygame.draw.rect(surface, self.LAMP_CORE, rect.inflate(-7, -6), border_radius=1)
        glow_circle(surface, center, 12, self.LAMP_GLOW)

    def _wall(self, floor, rect):
        shadow = pygame.Surface(floor.get_size(), pygame.SRCALPHA)
        for spread in range(10, 0, -1):
            pygame.draw.rect(
                shadow,
                (*self.SHADOW, 12 + (10 - spread) * 9),
                rect.move(5, 12).inflate(spread * 2, spread * 2),
                border_radius=5,
            )
        floor.blit(shadow, (0, 0))
        cut = min(7, rect.width // 5, rect.height // 5)
        pygame.draw.polygon(floor, self.WALL_BODY, chamfer(rect, cut))
        inner = rect.inflate(-14, -14).move(-1, -2)
        if inner.width <= 0 or inner.height <= 0:
            return
        # Textured top is clipped to its chamfer; body footprint remains the map rectangle.
        plate = pygame.Surface(inner.size, pygame.SRCALPHA)
        plate.blit(pygame.transform.smoothscale(material("steel"), inner.size), (0, 0))
        mask = pygame.Surface(inner.size, pygame.SRCALPHA)
        pygame.draw.polygon(mask, (255, 255, 255), chamfer(mask.get_rect(), 4))
        plate.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        floor.blit(plate, inner)
        pygame.draw.lines(floor, self.WALL_EDGE, False, chamfer(rect, cut)[:3], 1)
        pygame.draw.polygon(floor, self.WALL_BEVEL, chamfer(inner, 4), 1)
        if rect.height > 45:
            for x in (rect.left + 1, rect.right - 5):
                self._hazard(floor, pygame.Rect(x, rect.top + 16, 5, rect.height - 32))
        for x in (inner.left + 7, inner.right - 8):
            for y in (inner.top + 7, inner.bottom - 8):
                self._bolt(floor, (x, y), 4 if min(inner.size) > 40 else 2)

    def tank_underlay(self, surface, center, radius, color, analysis):
        r = round(radius)
        shadow = pygame.Surface((2 * r + 5, 2 * r), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (*self.SHADOW, 90), shadow.get_rect())
        surface.blit(shadow, (center[0] - r, center[1] - r + 5))

    def tank_body(self, sprite: TankSprite, tank, radius, color, recoil):
        unit = radius / DESIGN_RADIUS
        # The chassis turns with movement; the turret aims independently.
        heading = tank.velocity.normalized(tank.aim)
        body = chassis(color, sprite.px(34 * unit))
        sprite.blit_centered(
            pygame.transform.rotate(body, -math.degrees(math.atan2(heading.y, heading.x)) - 90)
        )
        aim, reach = tank.aim, 25 - recoil
        sprite.polygon(
            self.BARREL_SHADOW, [(0, -2.4), (reach, -2.4), (reach, 2.4), (0, 2.4)], aim, unit
        )
        sprite.polygon(
            self.BARREL, [(3, -1.5), (reach - 2, -1.5), (reach - 2, 1.5), (3, 1.5)], aim, unit
        )
        sprite.polygon(color, [(6, -2), (14, -2), (14, 2), (6, 2)], aim, unit)
        for ring, shade in (
            (8, self.HUB_SHADOW),
            (7, self.HUB_RIM),
            (6, color),
            (4, self.HUB_CORE),
            (2, self.HUB_CAP),
        ):
            sprite.circle(shade, ring * unit)

    def muzzle_flash(self, surface, tank, position, radius, recoil):
        muzzle = position + tank.aim * ((25 - recoil) * radius / DESIGN_RADIUS)
        glow_circle(surface, (round(muzzle.x), round(muzzle.y)), 6, self.theme.energy)

    def wreck(self, surface, tank, center, radius):
        sprite = TankSprite(radius)
        self.tank_body(sprite, tank, radius, self.WRECK, recoil=0)
        self.tank_underlay(surface, center, radius, self.WRECK, analysis=False)
        sprite.draw(surface, center)

    def projectile(self, surface, center, color, analysis):
        if not analysis:
            glow_circle(surface, center, 7, color)
        pygame.draw.circle(surface, tuple(round(c * 0.5) for c in color), center, 5)
        pygame.draw.circle(surface, color, center, 4)

    def particle(self, surface, position, velocity, life, color):
        end = position - velocity * min(0.04, life * 0.2)
        pygame.draw.aaline(surface, color, (end.x, end.y), (position.x, position.y))

    def trail(self, surface, first, second, color, index, count):
        shade = tuple(round(v * (0.5 + 0.5 * (index + 1) / count)) for v in color)
        pygame.draw.aaline(surface, shade, (first.x, first.y), (second.x, second.y))
