from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygame

from ricochet.display.styles import create_style
from ricochet.game.engine import GameEngine
from ricochet.game.maps import available_maps, load_map
from ricochet.runtime.loading import available_ais, controller_label
from ricochet.themes import DEFAULT_THEME, available_themes, get_theme


@dataclass(frozen=True)
class MenuSelection:
    map_id: str
    p1: str
    p2: str
    theme_id: str = DEFAULT_THEME


class Menu:
    def __init__(self, theme_id: str = DEFAULT_THEME):
        self.themes = available_themes()
        self.theme_id = get_theme(theme_id).id
        self.preview_cache = {}
        pygame.init()
        self.screen = pygame.display.set_mode((960, 740))
        pygame.display.set_caption("Ricochet / Theme selection")
        self.title = pygame.font.Font(None, 76)
        self.font = pygame.font.Font(None, 32)
        self.small = pygame.font.Font(None, 22)
        self.maps = available_maps()
        checkpoints = tuple(
            f"checkpoint:{p.resolve()}" for p in sorted(Path("runs").glob("*/best.zip"))
        )
        self.controllers = ("human",) + available_ais() + checkpoints
        default_bot = (
            self.controllers.index("bank_shot_ai") if "bank_shot_ai" in self.controllers else 1
        )
        self.indices = [self.maps.index("classic") if "classic" in self.maps else 0, 0, default_bot]
        self.indices.append(self.themes.index(theme_id))
        self.row = 0
        self.cards = [pygame.Rect(48 + i * 292, 460, 280, 128) for i in range(3)]
        self.cards.append(pygame.Rect(48, 610, 430, 64))
        self.start = pygame.Rect(520, 616, 380, 52)

    def _selection(self):
        p1, p2 = [self.controllers[i] for i in self.indices[1:3]]
        if p1 == p2 == "human":
            self.indices[2] = 1
            return None
        return MenuSelection(self.maps[self.indices[0]], p1, p2, self.themes[self.indices[3]])

    def run(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key in (pygame.K_UP, pygame.K_DOWN):
                        self.row = (self.row + (1 if event.key == pygame.K_DOWN else -1)) % 4
                    elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        self._cycle(self.row, 1 if event.key == pygame.K_RIGHT else -1)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        selected = self._selection()
                        if selected:
                            return selected
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.start.collidepoint(event.pos):
                        selected = self._selection()
                        if selected:
                            return selected
                    for i, rect in enumerate(self.cards):
                        if rect.collidepoint(event.pos):
                            self.row = i
                            self._cycle(i, -1 if event.pos[0] < rect.centerx else 1)
            self._draw()
            clock.tick(60)

    def _cycle(self, row, direction):
        count = len(self.maps if row == 0 else self.themes if row == 3 else self.controllers)
        self.indices[row] = (self.indices[row] + direction) % count

    def _draw(self):
        theme = get_theme(self.themes[self.indices[3]])
        self.screen.fill(theme.background)
        title = self.title.render("RICOCHET", True, theme.text)
        self.screen.blit(title, (48, 30))
        self.screen.blit(self.small.render(theme.description.upper(), True, theme.muted), (50, 104))
        map_id = self.maps[self.indices[0]]
        key = (map_id, theme.id)
        if key not in self.preview_cache:
            snapshot = GameEngine(load_map(map_id)).render_snapshot()
            style = create_style(theme)
            surface = style.floor(snapshot)
            for tank in snapshot.observation.tanks:
                style.tank(
                    surface,
                    tank,
                    tank.position,
                    snapshot.rules,
                    analysis=False,
                    flashes={},
                )
            self.preview_cache[key] = pygame.transform.smoothscale(surface, (460, 270))
        self.screen.blit(self.preview_cache[key], (250, 155))
        values = [self.maps[self.indices[0]].title()] + [
            controller_label(self.controllers[i]) for i in self.indices[1:3]
        ]
        values.append("Theme: " + theme.name)
        for i, (rect, label, value) in enumerate(
            zip(self.cards, ("ARENA", "PLAYER 01", "PLAYER 02", "THEME"), values, strict=True)
        ):
            color = theme.accent if i == self.row else theme.muted
            pygame.draw.rect(self.screen, theme.panel, rect, border_radius=8)
            pygame.draw.rect(self.screen, color, rect, width=2, border_radius=8)
            if i == 3:
                label = ""
            self.screen.blit(
                self.small.render(label, True, theme.muted), (rect.x + 18, rect.y + 16)
            )
            text = self.font.render(value, True, theme.text)
            if text.get_width() > rect.width - 44:
                text = pygame.transform.smoothscale(text, (rect.width - 44, 24))
            self.screen.blit(text, text.get_rect(center=rect.center))
            if i != 3:
                self.screen.blit(
                    self.small.render("<  choose  >", True, color),
                    (rect.x + 18, rect.bottom - 28),
                )
            else:
                self.screen.blit(self.small.render("<", True, color), (rect.x + 14, rect.y + 22))
                self.screen.blit(
                    self.small.render(">", True, color), (rect.right - 24, rect.y + 22)
                )
        pygame.draw.rect(self.screen, theme.players[0], self.start, border_radius=6)
        text = self.font.render("ENTER THE ARENA", True, theme.background)
        self.screen.blit(text, text.get_rect(center=self.start.center))
        self.screen.blit(
            self.small.render(
                "Click cards or use arrow keys / Enter starts / Esc quits", True, theme.muted
            ),
            (255, 697),
        )
        pygame.display.flip()
