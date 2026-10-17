import pygame

from ricochet.game.types import EventKind
from ricochet.resources import package_data


class Audio:
    def __init__(self):
        self.volume, self.muted = 0.35, False
        self.sounds = {}
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            for kind in (EventKind.FIRED, EventKind.BOUNCED, EventKind.BLOCKED, EventKind.HIT):
                with package_data("sounds", f"{kind}.wav").open("rb") as stream:
                    self.sounds[kind] = pygame.mixer.Sound(stream)
        except (pygame.error, OSError):
            self.sounds = {}

    def play(self, kind):
        if not self.muted and kind in self.sounds:
            self.sounds[kind].set_volume(self.volume)
            self.sounds[kind].play()

    def change_volume(self, amount):
        self.volume = max(0.0, min(1.0, self.volume + amount))
