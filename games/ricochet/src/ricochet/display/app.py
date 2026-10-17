from __future__ import annotations

from pathlib import Path

import pygame

from ricochet.display.human import HumanController
from ricochet.display.menu import Menu, MenuSelection
from ricochet.display.renderer import Renderer
from ricochet.game.config import RulesConfig
from ricochet.game.engine import GameEngine
from ricochet.game.maps import load_map
from ricochet.runtime.loading import controller_label
from ricochet.runtime.match import MatchSession, MatchSummary, safe_load_controller
from ricochet.runtime.replay import ReplayRecorder, load_replay, replay_engine, replay_steps
from ricochet.themes import DEFAULT_THEME


def _controller(name: str, viewport):
    return HumanController(viewport) if name == "human" else safe_load_controller(name)


def _label(name: str) -> str:
    return "Human" if name == "human" else controller_label(name)


def run_visual_match(
    selection: MenuSelection,
    seed: int = 0,
    rules: RulesConfig | None = None,
    record_path: Path | None = None,
) -> tuple[MatchSummary, bool]:
    active_rules = rules or RulesConfig()
    engine = GameEngine(load_map(selection.map_id, rules=active_rules), active_rules, seed)
    recorder = (
        ReplayRecorder(selection.map_id, active_rules, seed, (selection.p1, selection.p2))
        if record_path
        else None
    )
    renderer = Renderer(
        engine.render_snapshot(), (_label(selection.p1), _label(selection.p2)), selection.theme_id
    )
    session = MatchSession(
        engine,
        (
            _controller(selection.p1, renderer.viewport),
            _controller(selection.p2, renderer.viewport),
        ),
        (selection.p1, selection.p2),
        recorder,
    )
    if recorder is not None:
        recorder.initial_state_hash = engine.state_hash()
    clock = pygame.time.Clock()
    accumulator = 0.0
    previous = engine.render_snapshot()
    running = True
    while running and not session.finished:
        for event in pygame.event.get():
            renderer.handle_event(event)
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        if not running:
            break
        accumulator += min(clock.tick(60) / 1000.0, 0.1)
        while accumulator >= active_rules.dt and not session.finished:
            previous = engine.render_snapshot()
            result = session.step()
            renderer.ingest(result.events)
            accumulator -= active_rules.dt
        renderer.draw(engine.render_snapshot(), previous, accumulator / active_rules.dt)
    summary = session.summary()
    if recorder is not None and record_path is not None:
        recorder.write(record_path, summary)
    rematch = False
    if running and session.finished:
        winner_name = selection.p1 if summary.winner == 0 else selection.p2
        message = (
            "Invalid match"
            if summary.invalid
            else "Draw"
            if summary.winner is None
            else f"{_label(winner_name)} wins"
        )
        waiting = True
        while waiting:
            for event in pygame.event.get():
                renderer.handle_event(event)
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    rematch = True
                    waiting = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    waiting = False
            renderer.draw_result(engine.render_snapshot(), message)
            clock.tick(30)
    renderer.close()
    return summary, rematch


def run_menu(theme_id: str = DEFAULT_THEME) -> None:
    while True:
        selection = Menu(theme_id).run()
        if selection is None:
            pygame.quit()
            return
        theme_id = selection.theme_id
        rematch = True
        while rematch:
            _summary, rematch = run_visual_match(selection)


def run_visual_replay(path: Path, speed: float = 1.0, theme_id: str = DEFAULT_THEME) -> None:
    if speed <= 0:
        raise ValueError("replay speed must be positive")
    data = load_replay(path)
    engine = replay_engine(data)
    renderer = Renderer(engine.render_snapshot(), tuple(data["controllers"]), theme_id)
    iterator = iter(replay_steps(data))
    engine, _ = next(iterator)
    paused, running, ended = False, True, False
    clock = pygame.time.Clock()
    accumulator = 0.0
    while running:
        single = False
        for event in pygame.event.get():
            renderer.handle_event(event)
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_PERIOD:
                    single, paused = True, True
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_HOME):
                    target = (
                        0
                        if event.key == pygame.K_HOME
                        else engine.tick + (-150 if event.key == pygame.K_LEFT else 150)
                    )
                    iterator = iter(replay_steps(data, max(0, target)))
                    engine, _ = next(iterator)
                    renderer.trails.clear()
                    renderer.particles.clear()
                    renderer.dead.clear()
                    renderer.ripples.clear()
                    accumulator = 0.0
                    ended = False
                elif event.key == pygame.K_UP:
                    speed = min(16, speed * 2)
                elif event.key == pygame.K_DOWN:
                    speed = max(0.125, speed / 2)
        if not running:
            break
        elapsed = clock.tick(60) / 1000
        if not paused:
            accumulator += min(elapsed, 0.1) * speed
        if single:
            accumulator = engine.rules.dt
        while (not paused or single) and not ended and accumulator >= engine.rules.dt:
            try:
                engine, result = next(iterator)
                renderer.ingest(result.events)
            except StopIteration:
                ended, paused = True, True
            accumulator -= engine.rules.dt
            single = False
        label = "END" if ended else "PAUSED" if paused else "REPLAY"
        status = (
            f"{label} {speed:g}x  Space pause / . step / arrows seek,speed / "
            "Home restart / Esc back"
        )
        renderer.draw(engine.render_snapshot(), status=status)
    renderer.close()
