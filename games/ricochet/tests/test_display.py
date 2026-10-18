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


def test_viewport_transform():
    viewport = Viewport(960, 640, (1920, 1520))
    point = viewport.screen_to_world((200, (HUD_HEIGHT + 50) * 2))
    assert point.x == pytest.approx(100)
    assert point.y == pytest.approx(50)


def test_rendering_does_not_change_engine_or_hide_top_edge(monkeypatch):
    engine = GameEngine(load_map("classic"))
    other = GameEngine.from_state(engine.capture_state())
    renderer = Renderer(engine.render_snapshot(), ("One", "Two"))
    flips = []
    monkeypatch.setattr(pygame.display, "flip", lambda: flips.append(1))
    for _ in range(5):
        result = engine.step({0: ControlAction(fire=True)})
        other.step({0: ControlAction(fire=True)})
        renderer.ingest(result.events)
        renderer.draw(engine.render_snapshot())
        assert engine.state_hash() == other.state_hash()
    renderer.draw_result(engine.render_snapshot(), "Done")
    assert len(flips) == 6
    assert renderer.canvas.get_height() == 640 + 88 + 32
    assert renderer.viewport.screen_to_world((20, HUD_HEIGHT + 20)).y == 20
    renderer.close()


def test_audio_failure_is_nonfatal(monkeypatch):
    from ricochet.display.audio import Audio

    monkeypatch.setattr(pygame.mixer, "get_init", lambda: None)

    def fail():
        raise pygame.error("unavailable")

    monkeypatch.setattr(pygame.mixer, "init", fail)
    audio = Audio()
    audio.play("hit")
    assert audio.sounds == {}


def test_mouse_aim_uses_viewport_and_menu_mouse_selection(monkeypatch):
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
    original = menu.indices[0]
    events = [
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=(menu.cards[0].right - 10, menu.cards[0].centery)
        ),
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=menu.start.center),
    ]
    monkeypatch.setattr(pygame.event, "get", lambda: events)
    selection = menu.run()
    assert selection.map_id == menu.maps[(original + 1) % len(menu.maps)]
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
        engine = GameEngine(load_map(map_id))
        before = engine.state_hash()
        renderer = Renderer(engine.render_snapshot(), ("One", "Two"), theme_id)
        renderer.draw(engine.render_snapshot())
        renderer.handle_event(pygame.event.Event(pygame.VIDEORESIZE, w=1200, h=700))
        renderer.analysis = True
        renderer.draw(engine.render_snapshot())
        renderer.draw_result(engine.render_snapshot(), "Draw")
        assert engine.state_hash() == before
        renderer.close()
    assert len(flips) == 3 * len(available_maps())


def test_theme_menu_mouse_keyboard_and_cli(monkeypatch):
    from ricochet.display.menu import Menu
    from ricochet.runtime.cli import build_parser

    menu = Menu()
    menu._draw()
    events = [
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=(menu.cards[3].right - 5, menu.cards[3].centery)
        ),
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN),
    ]
    monkeypatch.setattr(pygame.event, "get", lambda: events)
    assert menu.run().theme_id == "robot-league"
    menu._draw()
    menu.row = 2
    events[:] = [
        pygame.event.Event(pygame.KEYDOWN, key=k)
        for k in (pygame.K_DOWN, pygame.K_LEFT, pygame.K_RETURN)
    ]
    assert menu.run().theme_id == "neon-circuit"
    pygame.display.quit()
    parser = build_parser()
    assert parser.parse_args(["match", "--theme", "robot-league"]).theme == "robot-league"
    assert parser.parse_args(["--theme", "robot-league", "match"]).theme == "robot-league"
    assert parser.parse_args(["match", "--theme", "neon-circuit"]).theme == "neon-circuit"
    replay = parser.parse_args(["replay", "r.json", "--theme", "robot-league"])
    assert replay.theme == "robot-league"
    with pytest.raises(SystemExit):
        parser.parse_args(["--theme", "robot-arena"])


def test_theme_catalog_is_headless_and_rejects_unknown():
    import subprocess
    import sys

    from ricochet.themes import get_theme

    with pytest.raises(ValueError, match="Unknown theme"):
        get_theme("missing")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from ricochet.themes import available_themes; "
            "assert available_themes(); assert 'pygame' not in sys.modules",
        ],
        check=True,
    )


def test_neon_shield_composes_behind_solid_tank():
    from dataclasses import replace

    from ricochet.display.styles import create_style
    from ricochet.themes import get_theme

    pygame.init()
    snapshot = GameEngine(load_map("classic")).render_snapshot()
    theme = get_theme("neon-circuit")
    tank = replace(snapshot.observation.tanks[0], shield_active=True)
    surface = pygame.Surface((960, 640))
    surface.fill(theme.background)
    create_style(theme).tank(
        surface,
        tank,
        tank.position,
        snapshot.rules,
        analysis=False,
        flashes={},
    )
    center = surface.get_at((round(tank.position.x), round(tank.position.y)))
    # A translucent shield fill drawn last would erase the solid turret sprite.
    assert center.g > 180 and center.b > 180
    pygame.display.quit()


def test_robot_league_shares_neon_hud_and_loads_transparent_chassis():
    from dataclasses import replace

    from ricochet.display.styles import create_style
    from ricochet.display.styles.robot_league import material
    from ricochet.themes import get_theme

    pygame.init()
    snapshot = GameEngine(load_map("classic")).render_snapshot()
    # Equal palettes must produce identical HUDs: prevents copied layouts drifting apart.
    theme = get_theme("neon-circuit")
    styles = (create_style(theme), create_style(replace(theme, arena_style="robot-league")))
    frames = []
    for style in styles:
        surface = pygame.Surface((960, HUD_HEIGHT))
        surface.fill(theme.panel)
        style.hud(surface, snapshot, ("One", "Two"))
        frames.append(pygame.image.tobytes(surface, "RGB"))
    assert frames[0] == frames[1]
    chassis = material("chassis")
    assert chassis.get_at((0, 0)).a <= 2
    assert chassis.get_at(chassis.get_rect().center).a > 250
    assert material("concrete").get_width() > 0
    assert material("steel").get_width() > 0
    pygame.display.quit()


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
