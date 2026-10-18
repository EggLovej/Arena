"""Replays of the standalone game: every action and state hash, with seeking for the viewer.

The Arena records its own replays (``platform/arena``); these are for the ``ricochet`` command.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ricochet.game.config import RulesConfig
from ricochet.game.engine import GameEngine
from ricochet.game.types import ControlAction, StepResult
from ricochet.runtime.loading import default_ais_directory
from ricochet.runtime.match import MatchSummary

REPLAY_SCHEMA_VERSION = 2
ENGINE_VERSION = "2.0.0"


@dataclass
class ReplayRecorder:
    map_id: str
    rules: RulesConfig
    seed: int
    controllers: tuple[str, str]
    initial_state_hash: str = ""
    initial_state: dict | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    snapshots: dict[str, dict] = field(default_factory=dict)
    forfeit_player: int | None = None

    def record_step(
        self,
        actions: dict[int, ControlAction],
        result: StepResult,
        engine: GameEngine | None = None,
    ) -> None:
        self.steps.append(
            {
                "actions": [actions[p].to_dict() for p in (0, 1)],
                "events": [e.to_dict() for e in result.events],
                "state_hash": result.state_hash,
            }
        )
        if engine is not None and engine.tick % 300 == 0:
            self.snapshots[str(engine.tick)] = engine.capture_state()

    def record_forfeit(self, player_id: int) -> None:
        self.forfeit_player = player_id

    def to_dict(self, summary: MatchSummary) -> dict[str, Any]:
        identities = []
        for name in self.controllers:
            path = (
                Path(name.split(":", 1)[1])
                if name.startswith("checkpoint:")
                else default_ais_directory() / f"{name}.py"
            )
            identities.append(
                {
                    "name": name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                    if path.is_file()
                    else None,
                }
            )
        return {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "engine_version": ENGINE_VERSION,
            "map_id": self.map_id,
            "rules": self.rules.to_dict(),
            "seed": self.seed,
            "controllers": list(self.controllers),
            "controller_identities": identities,
            "initial_state": self.initial_state,
            "initial_state_hash": self.initial_state_hash,
            "steps": self.steps,
            "snapshots": self.snapshots,
            "forfeit_player": self.forfeit_player,
            "summary": summary.to_dict(),
        }

    def write(self, path: Path, summary: MatchSummary) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(summary), indent=2) + "\n")


def load_replay(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ValueError("incompatible replay schema version")
    if data.get("engine_version") != ENGINE_VERSION:
        raise ValueError("incompatible replay engine version")
    return data


def replay_engine(data: dict) -> GameEngine:
    # Construct entirely from the embedded state; installed map files are irrelevant.
    engine = GameEngine.from_state(data["initial_state"])
    if engine.state_hash() != data["initial_state_hash"]:
        raise ValueError("replay initial state hash does not match")
    return engine


def replay_steps(data: dict, start_tick: int = 0) -> Iterator[tuple[GameEngine, StepResult]]:
    engine = replay_engine(data)
    start_tick = max(0, min(start_tick, len(data["steps"])))
    candidates = [int(t) for t in data.get("snapshots", {}) if int(t) <= start_tick]
    if candidates:
        tick = max(candidates)
        engine.restore_state(data["snapshots"][str(tick)])
        if engine.state_hash() != data["steps"][tick - 1]["state_hash"]:
            raise ValueError("replay snapshot hash mismatch")
    if engine.tick == start_tick:
        yield engine, engine.snapshot()
    for index in range(engine.tick, len(data["steps"])):
        recorded = data["steps"][index]
        result = engine.step({p: ControlAction.from_dict(recorded["actions"][p]) for p in (0, 1)})
        if result.state_hash != recorded["state_hash"]:
            raise ValueError(f"replay diverged at tick {index + 1}")
        if engine.tick >= start_tick:
            yield engine, result
    forfeiter = data.get("forfeit_player")
    if forfeiter is not None:
        if forfeiter == -1:
            engine.terminated, engine.winner = True, None
        else:
            engine.force_forfeit(int(forfeiter))
    if engine.state_hash() != data["summary"]["final_state_hash"]:
        raise ValueError("replay final state hash does not match")
    yield engine, engine.snapshot()


def verify_replay(path: Path) -> dict:
    data = load_replay(path)
    for _ in replay_steps(data):
        pass
    return data["summary"]
