"""Capture a staged, deterministic visual fixture (not a simulated match).

Run from the repo root: PYTHONPATH=src python scripts/capture_theme_preview.py
Uses real renderer, map and rules; arranges tanks/shots to compare with the proposal.
"""

import os
from collections import deque
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from ricochet.display.renderer import Renderer
from ricochet.game.engine import GameEngine
from ricochet.game.maps import load_map
from ricochet.game.types import ProjectileView, Vec2
from ricochet.themes import available_themes


def capture():
    snapshot = GameEngine(load_map("classic")).render_snapshot()
    obs = snapshot.observation
    tanks = (
        replace(
            obs.tanks[0],
            position=Vec2(270, 480),
            aim=Vec2(0.707, -0.707),
            energy=69,
            shield_active=True,
        ),
        replace(
            obs.tanks[1], position=Vec2(730, 145), aim=Vec2(0.707, -0.707), energy=51, health=2
        ),
    )
    shots = (
        ProjectileView(1, 0, Vec2(360, 390), Vec2(300, -300), False, 1),
        ProjectileView(2, 1, Vec2(865, 61), Vec2(300, -200), False, 1),
    )
    snapshot = replace(
        snapshot,
        observation=replace(obs, tanks=tanks, projectiles=shots, time_seconds=22.6, tick=678),
    )
    output = Path("docs/screenshots")
    output.mkdir(parents=True, exist_ok=True)
    for theme in available_themes():
        renderer = Renderer(snapshot, ("Pursuit", "Bank Shot"), theme)
        renderer.last_tick = snapshot.observation.tick
        for shot in shots:
            renderer.trails[shot.id] = deque(
                [shot.position - shot.velocity * (i * 0.032) for i in range(6, -1, -1)], maxlen=7
            )
        renderer.draw(snapshot)
        pygame.image.save(renderer.canvas, output / f"{theme}_preview.png")
        renderer.close()


if __name__ == "__main__":
    capture()
