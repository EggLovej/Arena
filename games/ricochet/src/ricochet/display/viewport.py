from dataclasses import dataclass

from ricochet.game.types import Vec2

HUD_HEIGHT = 88
FOOTER_HEIGHT = 32


@dataclass
class Viewport:
    width: float
    height: float
    window_size: tuple[int, int]

    @property
    def scale(self):
        return min(
            self.window_size[0] / self.width,
            self.window_size[1] / (self.height + HUD_HEIGHT + FOOTER_HEIGHT),
        )

    @property
    def offset(self):
        return (
            (self.window_size[0] - self.width * self.scale) / 2,
            (self.window_size[1] - (self.height + HUD_HEIGHT + FOOTER_HEIGHT) * self.scale) / 2,
        )

    def screen_to_world(self, point):
        x, y = self.offset
        return Vec2((point[0] - x) / self.scale, (point[1] - y) / self.scale - HUD_HEIGHT)
