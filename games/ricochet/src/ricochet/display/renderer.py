from __future__ import annotations

import math
import random
from collections import deque

import pygame

from ricochet.display import controls
from ricochet.display.audio import Audio
from ricochet.display.styles import create_style
from ricochet.display.viewport import FOOTER_HEIGHT, HUD_HEIGHT, Viewport
from ricochet.game.types import EventKind, RenderSnapshot, Vec2
from ricochet.themes import DEFAULT_THEME, Theme, get_theme


class Renderer:
    """Pure presentation: consumes frozen snapshots/events, never the engine."""

    def __init__(
        self,
        snapshot: RenderSnapshot,
        names: tuple[str, str],
        theme: str | Theme = DEFAULT_THEME,
    ):
        self.theme = get_theme(theme) if isinstance(theme, str) else theme
        self.style = create_style(self.theme)
        pygame.init()
        obs = snapshot.observation
        self.logical_size = (int(obs.width), int(obs.height) + HUD_HEIGHT + FOOTER_HEIGHT)
        self.screen = pygame.display.set_mode(self.logical_size, pygame.RESIZABLE)
        pygame.display.set_caption(f"Ricochet / {self.theme.name}")
        self.canvas = pygame.Surface(self.logical_size)
        self.arena = pygame.Surface((int(obs.width), int(obs.height)))
        self.viewport = Viewport(obs.width, obs.height, self.screen.get_size())
        self.names = names
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 62)
        self.audio = Audio()
        self.analysis = False
        self.random = random.Random(42)
        self.particles = []
        self.ripples = []
        self.bounces = {}
        self.trails = {}
        self.flashes = {}
        self.dead = set()
        self.clock = pygame.time.Clock()
        self.last_tick = -1
        self.floor = self.style.floor(snapshot)

    def ingest(self, events):
        self.bounces = {}
        for event in events:
            if event.kind == EventKind.BOUNCED and event.position is not None:
                self.bounces[event.projectile_id] = event.position
            if event.kind == EventKind.BLOCKED and event.position is not None and not self.analysis:
                self.ripples.append((event.position, 0.25))
            self.audio.play(event.kind)
            if event.kind in (EventKind.FIRED, EventKind.HIT) and event.player_id is not None:
                self.flashes[(event.kind, event.player_id)] = 0.12
            if event.position is None or self.analysis:
                continue
            color = (
                self.theme.energy
                if event.kind == EventKind.BOUNCED
                else self.theme.shield
                if event.kind == EventKind.BLOCKED
                else self.theme.players[event.player_id]
                if event.player_id is not None
                else self.theme.text
            )
            count = {
                EventKind.HIT: 14,
                EventKind.BOUNCED: 10,
                EventKind.BLOCKED: 12,
                EventKind.FIRED: 4,
                EventKind.SHIELD_ACTIVATED: 10,
            }.get(event.kind, 0)
            for _ in range(count):
                angle = self.random.random() * math.tau
                speed = self.random.uniform(30, 100)
                self.particles.append(
                    [
                        event.position,
                        Vec2(math.cos(angle), math.sin(angle)) * speed,
                        self.random.uniform(0.12, 0.35),
                        color,
                    ]
                )
        self.particles = self.particles[-300:]

    def handle_event(self, event):
        if event.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode(
                (max(480, event.w), max(380, event.h)), pygame.RESIZABLE
            )
            self.viewport.window_size = self.screen.get_size()
        if event.type == pygame.KEYDOWN:
            if event.key == controls.TOGGLE_ANALYSIS:
                self.analysis = not self.analysis
                self.particles.clear()
            elif event.key == controls.TOGGLE_MUTE:
                self.audio.muted = not self.audio.muted
            elif event.key in controls.VOLUME_UP:
                self.audio.change_volume(0.1)
            elif event.key == controls.VOLUME_DOWN:
                self.audio.change_volume(-0.1)

    def _compose(self, snapshot, previous=None, alpha=1.0, status=""):
        obs, rules = snapshot.observation, snapshot.rules
        dt = min(self.clock.tick() / 1000, 0.05)
        self.flashes = {k: v - dt for k, v in self.flashes.items() if v > dt}
        self.arena.blit(self.floor, (0, 0))
        old = previous.observation if previous else obs
        for shot in obs.projectiles:
            prior = next((p for p in old.projectiles if p.id == shot.id), None)
            display_position = shot.position
            if prior is not None:
                impact = self.bounces.get(shot.id)
                if impact is not None and prior.bounced != shot.bounced:
                    first_length = (impact - prior.position).length()
                    total = first_length + (shot.position - impact).length()
                    split = first_length / total if total else 0
                    if alpha < split:
                        fraction = alpha / split
                        display_position = prior.position * (1 - fraction) + impact * fraction
                    else:
                        fraction = (alpha - split) / (1 - split) if split < 1 else 1
                        display_position = impact * (1 - fraction) + shot.position * fraction
                elif prior.bounced == shot.bounced:
                    display_position = prior.position * (1 - alpha) + shot.position * alpha
            trail = self.trails.setdefault(shot.id, deque(maxlen=6))
            if obs.tick != self.last_tick:
                prior = next((p for p in old.projectiles if p.id == shot.id), None)
                if prior and prior.bounced != shot.bounced:
                    trail.clear()
                trail.append(shot.position)
            color = self.theme.players[shot.owner_id]
            if not self.analysis and len(trail) > 1:
                points = list(trail)
                for index, (first, second) in enumerate(zip(points, points[1:], strict=False)):
                    self.style.trail(self.arena, first, second, color, index, len(points))
            center = (round(display_position.x), round(display_position.y))
            self.style.projectile(self.arena, center, color, self.analysis)
            pygame.draw.circle(self.arena, self.theme.text, center, 3)
            if shot.bounced:
                pygame.draw.circle(self.arena, self.theme.energy, center, 9, 1)
            if self.analysis:
                end = shot.position + shot.velocity * 0.25
                pygame.draw.line(self.arena, color, center, (end.x, end.y), 1)
        self.trails = {p.id: self.trails[p.id] for p in obs.projectiles}
        for tank in obs.tanks:
            before = old.tank(tank.player_id)
            position = before.position * (1 - alpha) + tank.position * alpha
            self.style.tank(
                self.arena,
                tank,
                position,
                rules,
                analysis=self.analysis,
                flashes=self.flashes,
            )
            if tank.health == 0 and tank.player_id not in self.dead:
                self.dead.add(tank.player_id)
                if not self.analysis:
                    for _ in range(35):
                        angle = self.random.random() * math.tau
                        self.particles.append(
                            [
                                tank.position,
                                Vec2(math.cos(angle), math.sin(angle)) * 130,
                                0.5,
                                self.theme.energy,
                            ]
                        )
        survivors = []
        for position, velocity, life, color in self.particles:
            if life > dt:
                position = position + velocity * dt
                self.style.particle(self.arena, position, velocity, life, color)
                survivors.append([position, velocity, life - dt, color])
        self.particles = survivors[-300:]
        self.ripples = [(position, life - dt) for position, life in self.ripples if life > dt][-24:]
        for position, life in self.ripples:
            radius = round(4 + (1 - life / 0.25) * 18)
            pygame.draw.circle(
                self.arena, self.theme.shield, (round(position.x), round(position.y)), radius, 1
            )
        self.canvas.fill(self.theme.panel)
        self.canvas.blit(self.arena, (0, HUD_HEIGHT))
        self.style.hud(self.canvas, snapshot, self.names)
        footer = status or controls.HINT
        if self.analysis:
            footer = (
                f"ANALYSIS    Tick {obs.tick}    {len(obs.projectiles)} projectiles    " + status
            )
        self.style.footer(
            self.canvas, footer, obs.height + HUD_HEIGHT, status=bool(status) or self.analysis
        )
        self.last_tick = obs.tick

    def _present(self):
        self.viewport.window_size = self.screen.get_size()
        scale = self.viewport.scale
        size = (round(self.logical_size[0] * scale), round(self.logical_size[1] * scale))
        self.screen.fill(self.theme.background)
        self.screen.blit(pygame.transform.smoothscale(self.canvas, size), self.viewport.offset)
        pygame.display.flip()

    def draw(self, snapshot, previous=None, alpha=1.0, status=""):
        self._compose(snapshot, previous, alpha, status)
        self._present()

    def draw_result(self, snapshot, message):
        self._compose(snapshot)
        overlay = pygame.Surface(self.logical_size, pygame.SRCALPHA)
        overlay.fill((*self.theme.background, 220))
        self.canvas.blit(overlay, (0, 0))
        center = self.canvas.get_rect().center
        title = self.large_font.render(message, True, self.theme.text)
        prompt = self.font.render("Enter: rematch    Esc: menu", True, self.theme.muted)
        self.canvas.blit(title, title.get_rect(center=(center[0], center[1] - 25)))
        self.canvas.blit(prompt, prompt.get_rect(center=(center[0], center[1] + 35)))
        self._present()

    def close(self):
        pygame.display.quit()
