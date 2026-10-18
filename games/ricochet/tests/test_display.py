import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from ricochet.display.renderer import Renderer
from ricochet.display.viewport import HUD_HEIGHT, Viewport
from ricochet.game.engine import GameEngine
from ricochet.game.maps import load_map
from ricochet.game.types import ControlAction


def test_audio_failure_is_nonfatal(monkeypatch):
    from ricochet.display.audio import Audio

    monkeypatch.setattr(pygame.mixer, "get_init", lambda: None)

    def fail():
        raise pygame.error("unavailable")

    monkeypatch.setattr(pygame.mixer, "init", fail)
    audio = Audio()
    audio.play("hit")
    assert audio.sounds == {}


def test_mouse_aim_uses_viewport_and_menu_takes_mouse_and_keys(monkeypatch):
    from ricochet.display.human import HumanController
    from ricochet.display.menu import Menu
    from ricochet.game.types import Vec2

    engine = GameEngine(load_map("classic"))
    viewport = Viewport(960, 640, (1920, 1520))
    controller = HumanController(viewport)
    monkeypatch.setattr(
        pygame.key,
        "get_pressed",
        lambda: {
            pygame.K_d: False,
            pygame.K_a: False,
            pygame.K_s: False,
            pygame.K_w: False,
            pygame.K_SPACE: False,
        },
    )
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (440, (HUD_HEIGHT + 320) * 2))
    monkeypatch.setattr(pygame.mouse, "get_pressed", lambda n: (False, False, False))
    assert controller.act(engine.observation_for(0)).aim == Vec2(100, 0)
    menu = Menu()
    menu._draw()
    original = menu.indices[0]
    events = [
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(card.right - 10, card.centery))
        for card in (menu.cards[0], menu.cards[3])  # The next map, then the next theme.
    ]
    events.append(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=menu.start.center))
    monkeypatch.setattr(pygame.event, "get", lambda: events)
    selection = menu.run()
    assert selection.map_id == menu.maps[(original + 1) % len(menu.maps)]
    assert selection.theme_id == "robot-league"
    menu._draw()
    menu.row = 2
    events[:] = [
        pygame.event.Event(pygame.KEYDOWN, key=k)
        for k in (pygame.K_DOWN, pygame.K_LEFT, pygame.K_RETURN)
    ]
    assert menu.run().theme_id == "neon-circuit"
    pygame.display.quit()


def test_replay_controls_and_visual_match_through_app(tmp_path, monkeypatch):
    from ricochet.display.app import run_visual_match, run_visual_replay
    from ricochet.display.menu import MenuSelection
    from ricochet.game.config import RulesConfig
    from ricochet.runtime.match import run_headless

    path = tmp_path / "replay.json"
    run_headless("direct_ai", "bank_shot_ai", rules=RulesConfig(max_seconds=0.1), record_path=path)
    batches = iter(
        [
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_PERIOD)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_HOME)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN)],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)],
        ]
    )
    monkeypatch.setattr(pygame.event, "get", lambda: next(batches, []))
    run_visual_replay(path)
    calls = 0

    def end_result():
        nonlocal calls
        calls += 1
        return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)] if calls > 8 else []

    monkeypatch.setattr(pygame.event, "get", end_result)
    summary, _ = run_visual_match(
        MenuSelection("classic", "direct_ai", "bank_shot_ai"), rules=RulesConfig(max_seconds=0.1)
    )
    assert summary.ticks == 3


@pytest.mark.parametrize("theme_id", ["neon-circuit", "robot-league"])
def test_themes_render_all_maps_without_mutating_state(theme_id, monkeypatch):
    from ricochet.game.maps import available_maps

    flips = []
    monkeypatch.setattr(pygame.display, "flip", lambda: flips.append(1))
    for map_id in available_maps():
        engine, unrendered = GameEngine(load_map(map_id)), GameEngine(load_map(map_id))
        renderer = Renderer(engine.render_snapshot(), ("One", "Two"), theme_id)
        # The HUD sits above the arena and hides none of it.
        assert renderer.viewport.screen_to_world((20, HUD_HEIGHT + 20)).y == 20
        for _ in range(3):
            renderer.ingest(engine.step({0: ControlAction(fire=True)}).events)
            unrendered.step({0: ControlAction(fire=True)})
            renderer.draw(engine.render_snapshot())
        renderer.handle_event(pygame.event.Event(pygame.VIDEORESIZE, w=1200, h=700))
        renderer.analysis = True
        renderer.draw(engine.render_snapshot())
        renderer.draw_result(engine.render_snapshot(), "Draw")
        assert engine.state_hash() == unrendered.state_hash()
        renderer.close()
    assert len(flips) == 5 * len(available_maps())  # One per frame.


def test_new_style_needs_only_floor_and_tank_body():
    from dataclasses import replace

    from ricochet.display.styles import BaseStyle
    from ricochet.themes import get_theme

    class PlainStyle(BaseStyle):
        def floor(self, snapshot):
            obs = snapshot.observation
            surface = pygame.Surface((int(obs.width), int(obs.height)))
            surface.fill(self.theme.background)
            return surface

        def tank_body(self, sprite, tank, radius, color, recoil):
            sprite.circle(color, radius)

    with pytest.raises(TypeError):
        BaseStyle(get_theme())
    pygame.init()
    snapshot = GameEngine(load_map("classic")).render_snapshot()
    style = PlainStyle(get_theme())
    surface = style.floor(snapshot)
    shielded = replace(snapshot.observation.tanks[0], shield_active=True, shield_ticks_remaining=3)
    for tank in (shielded, replace(snapshot.observation.tanks[1], health=0)):
        style.tank(
            surface,
            tank,
            tank.position,
            snapshot.rules,
            analysis=True,
            flashes={("hit", tank.player_id): 0.1},
        )
    hud = pygame.Surface((960, HUD_HEIGHT))
    style.hud(hud, snapshot, ("One", "Two"))
    style.footer(hud, "status line", 0, status=True)
    center = surface.get_at((round(shielded.position.x), round(shielded.position.y)))
    # The body covers the translucent shield field (downsampling softens it slightly).
    assert all(abs(a - b) <= 8 for a, b in zip(center[:3], get_theme().players[0], strict=True))
    pygame.display.quit()
