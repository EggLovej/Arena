from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ricochet.bot import Controller
from ricochet.game.config import RulesConfig
from ricochet.game.engine import GameEngine
from ricochet.game.maps import jitter_spawns, load_map
from ricochet.game.types import EventKind
from ricochet.runtime.match import ControllerSlot, FailedController, safe_load_controller
from ricochet.training.encoding import (
    ACTION_SIZE,
    OBSERVATION_SIZE,
    check_capacity,
    decode_action,
    encode_observation,
)


class RicochetEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        map_id="classic",
        opponent: str | Callable[[], Controller] = "direct_ai",
        seat_mode: Literal["random", 0, 1] = "random",
        rules=None,
        maps: tuple[str, ...] | None = None,
        jitter_probability=0.0,
        max_episode_steps: int | None = None,
        verification=False,
    ):
        super().__init__()
        if seat_mode not in ("random", 0, 1):
            raise ValueError("invalid seat mode")
        self.map_id, self.maps = map_id, maps or (map_id,)
        self.opponent_source, self.seat_mode = opponent, seat_mode
        self.rules = rules or RulesConfig()
        check_capacity(self.rules)
        if not 0 <= jitter_probability <= 1:
            raise ValueError("invalid jitter probability")
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        self.jitter_probability, self.max_episode_steps = jitter_probability, max_episode_steps
        self.verification = verification
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.engine = self.opponent = None
        self.opponent_slot: ControllerSlot | None = None
        self.player_id = 0
        self.episode_seed = 0
        self._done = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.player_id = (
            int(self.np_random.integers(0, 2))
            if self.seat_mode == "random"
            else int(self.seat_mode)
        )
        self.map_id = self.maps[int(self.np_random.integers(len(self.maps)))]
        jitter = options.get("jitter", self.np_random.random() < self.jitter_probability)
        definition = load_map(self.map_id, rules=self.rules)
        if jitter:
            definition = jitter_spawns(definition, self.rules, self.np_random)
        self.engine = GameEngine(definition, self.rules, self.episode_seed, self.verification)
        if not isinstance(self.opponent_source, str):
            try:
                self.opponent = self.opponent_source()
            except Exception as error:
                self.opponent = FailedController(error)
        elif self.opponent is None:
            self.opponent = safe_load_controller(self.opponent_source)
        self.opponent_slot = ControllerSlot(self.opponent, self.rules.max_bot_faults)
        self.opponent_slot.reset(self.engine.context_for(1 - self.player_id))
        self._done = False
        self.last_components = {"damage_dealt": 0.0, "damage_taken": 0.0, "outcome": 0.0}
        return encode_observation(
            self.engine.observation_for(self.player_id), self.rules
        ), self._info(())

    def step(self, action):
        if self.engine is None or self._done:
            raise RuntimeError("reset() must be called before stepping a new episode")
        learner_action = decode_action(action)
        opponent_id = 1 - self.player_id
        opponent_action = self.opponent_slot.act(self.engine.observation_for(opponent_id))
        before_health = [self.engine.tanks[p].health for p in (0, 1)]
        if self.opponent_slot.failed:
            self.engine.force_forfeit(opponent_id)
            result = self.engine.snapshot()
        else:
            result = self.engine.step(
                {self.player_id: learner_action, opponent_id: opponent_action}
            )
        taken = before_health[self.player_id] - self.engine.tanks[self.player_id].health
        owned_hits = sum(
            e.kind == EventKind.HIT and e.player_id == opponent_id and e.value == self.player_id
            for e in result.events
        )
        dealt = min(owned_hits, before_health[opponent_id] - self.engine.tanks[opponent_id].health)
        outcome = 0.0
        if result.terminated and result.winner is not None:
            outcome = 1.0 if result.winner == self.player_id else -1.0
        self.last_components = {
            "damage_dealt": 0.1 * dealt,
            "damage_taken": -0.1 * taken,
            "outcome": outcome,
        }
        truncated = bool(
            not result.terminated
            and self.max_episode_steps is not None
            and self.engine.tick >= self.max_episode_steps
        )
        self._done = result.terminated or truncated
        observation = encode_observation(result.observation_for(self.player_id), self.rules)
        return (
            observation,
            sum(self.last_components.values()),
            result.terminated,
            truncated,
            self._info(result.events, result.state_hash),
        )

    def _info(self, events, state_hash="") -> dict[str, Any]:
        return {
            "events": [e.to_dict() for e in events],
            "winner": self.engine.winner,
            "state_hash": state_hash or (self.engine.state_hash() if self.verification else ""),
            "opponent_faults": self.opponent_slot.stats.faults,
            "opponent_name": getattr(
                self.opponent,
                "last_name",
                self.opponent_source if isinstance(self.opponent_source, str) else "factory",
            ),
            "health": [self.engine.tanks[p].health for p in (0, 1)],
            "player_id": self.player_id,
            "episode_seed": self.episode_seed,
            "map_id": self.map_id,
            "reward_components": self.last_components,
        }
