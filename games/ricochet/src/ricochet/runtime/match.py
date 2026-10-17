from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ricochet.api import Controller, create_noop_action, validate_action
from ricochet.game.config import RulesConfig
from ricochet.game.engine import GameEngine
from ricochet.game.maps import load_map
from ricochet.game.types import (
    PLAYERS,
    ControlAction,
    GameObservation,
    MapDefinition,
    MatchContext,
    StepResult,
)
from ricochet.runtime.loading import load_controller

if TYPE_CHECKING:
    from ricochet.runtime.replay import ReplayRecorder


@dataclass
class ControllerStats:
    faults: int = 0
    calls: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.calls if self.calls else 0.0


class ControllerSlot:
    """One seat: validates actions, counts faults and times calls for any controller."""

    def __init__(self, controller: Controller, max_faults: int) -> None:
        self.controller = controller
        self.max_faults = max_faults
        self.stats = ControllerStats()

    @property
    def failed(self) -> bool:
        return self.stats.faults >= self.max_faults

    def reset(self, context: MatchContext) -> None:
        try:
            self.controller.reset(context)
        except Exception as error:  # A startup failure is an immediate forfeit.
            self.stats.faults = self.max_faults
            self.stats.errors.append(f"startup: {type(error).__name__}: {error}")

    def act(self, observation: GameObservation) -> ControlAction:
        if self.failed:
            return create_noop_action()
        stats = self.stats
        started = time.perf_counter()
        try:
            return validate_action(self.controller.act(observation))
        except Exception as error:  # Local plugins may fail in arbitrary ways.
            stats.faults += 1
            stats.errors.append(f"{type(error).__name__}: {error}")
            return create_noop_action()
        finally:
            elapsed = time.perf_counter() - started
            stats.calls += 1
            stats.total_seconds += elapsed
            stats.max_seconds = max(stats.max_seconds, elapsed)


@dataclass(frozen=True)
class MatchSummary:
    map_id: str
    seed: int
    controllers: tuple[str, str]
    winner: int | None
    draw: bool
    ticks: int
    seconds: float
    health: tuple[int, int]
    faults: tuple[int, int]
    mean_action_ms: tuple[float, float]
    max_action_ms: tuple[float, float]
    final_state_hash: str
    forfeit_player: int | None = None
    invalid: bool = False
    errors: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MatchSession:
    def __init__(
        self,
        engine: GameEngine,
        controllers: tuple[Controller, Controller],
        controller_names: tuple[str, str],
        recorder: ReplayRecorder | None = None,
    ) -> None:
        self.engine = engine
        self.controller_names = controller_names
        self.slots = tuple(
            ControllerSlot(controller, engine.rules.max_bot_faults) for controller in controllers
        )
        self.recorder = recorder
        self.forfeit_player: int | None = None
        self.invalid = False
        if recorder is not None:
            recorder.initial_state = engine.capture_state()
        for player_id in PLAYERS:
            self.slots[player_id].reset(engine.context_for(player_id))

    @property
    def finished(self) -> bool:
        return self.engine.terminated or self.engine.timed_out

    def step(self) -> StepResult:
        if self.finished:
            raise RuntimeError("cannot step a finished match")
        actions = {
            player_id: self.slots[player_id].act(self.engine.observation_for(player_id))
            for player_id in PLAYERS
        }
        failed = [player_id for player_id in PLAYERS if self.slots[player_id].failed]
        if len(failed) == len(PLAYERS):
            self.invalid = True
            self.engine.terminated = True
            self.engine.winner = None
            if self.recorder is not None:
                self.recorder.record_forfeit(-1)
            return self.engine.snapshot()
        if failed:
            self.forfeit_player = failed[0]
            self.engine.force_forfeit(failed[0])
            if self.recorder is not None:
                self.recorder.record_forfeit(failed[0])
            return self.engine.snapshot()
        result = self.engine.step(actions)
        if self.recorder is not None:
            self.recorder.record_step(actions, result, self.engine)
        return result

    def summary(self) -> MatchSummary:
        engine = self.engine
        stats = [slot.stats for slot in self.slots]
        return MatchSummary(
            map_id=engine.map_definition.id,
            seed=engine.seed,
            controllers=self.controller_names,
            winner=engine.winner,
            draw=engine.winner is None,
            ticks=engine.tick,
            seconds=engine.time_seconds,
            health=(engine.tanks[0].health, engine.tanks[1].health),
            faults=(stats[0].faults, stats[1].faults),
            mean_action_ms=tuple(seat.mean_seconds * 1000 for seat in stats),
            max_action_ms=tuple(seat.max_seconds * 1000 for seat in stats),
            final_state_hash=engine.state_hash(),
            forfeit_player=self.forfeit_player,
            invalid=self.invalid,
            errors=tuple(tuple(seat.errors) for seat in stats),
        )


def resolve_map(map_id: str | MapDefinition, rules: RulesConfig) -> MapDefinition:
    return map_id if isinstance(map_id, MapDefinition) else load_map(map_id, rules=rules)


def create_session(
    p1_name: str,
    p2_name: str,
    map_id: str | MapDefinition = "classic",
    seed: int = 0,
    rules: RulesConfig | None = None,
    recorder: ReplayRecorder | None = None,
) -> MatchSession:
    active_rules = rules or RulesConfig()
    # Per-tick state hashes are only needed to record a verifiable replay.
    engine = GameEngine(
        resolve_map(map_id, active_rules), active_rules, seed, verification=recorder is not None
    )
    return MatchSession(
        engine,
        (safe_load_controller(p1_name), safe_load_controller(p2_name)),
        (p1_name, p2_name),
        recorder,
    )


def run_headless(
    p1_name: str,
    p2_name: str,
    map_id: str | MapDefinition = "classic",
    seed: int = 0,
    rules: RulesConfig | None = None,
    record_path: Path | None = None,
    on_step: Callable[[StepResult], None] | None = None,
) -> MatchSummary:
    active_rules = rules or RulesConfig()
    definition = resolve_map(map_id, active_rules)
    recorder = None
    if record_path is not None:
        from ricochet.runtime.replay import ReplayRecorder

        recorder = ReplayRecorder(definition.id, active_rules, seed, (p1_name, p2_name))
    session = create_session(p1_name, p2_name, definition, seed, active_rules, recorder)
    if recorder is not None:
        recorder.initial_state_hash = session.engine.state_hash()
    while not session.finished:
        result = session.step()
        if on_step is not None:
            on_step(result)
    summary = session.summary()
    if recorder is not None and record_path is not None:
        recorder.write(record_path, summary)
    return summary


class FailedController:
    def __init__(self, error: Exception):
        self.error = error

    def reset(self, context):
        raise self.error

    def act(self, observation):
        return create_noop_action()


def safe_load_controller(name: str) -> Controller:
    try:
        return load_controller(name)
    except Exception as error:
        return FailedController(error)
