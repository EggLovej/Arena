"""``NeuralAgent`` without PyTorch: the same network, worked out with numpy.

Where PyTorch is not installed, ``ais/neural_ai.py`` and ``ais/beast_ai.py``
play through this agent instead. The Arena website runs its bots in the
visitor's browser, where PyTorch cannot run, so this is how people play
against Instinct and Apex there. It makes its choices exactly as
``NeuralAgent`` does (ml/play/numpy_model.py has the network), drawing
them from its own random numbers.

Only ``network:`` players play without PyTorch; the look-ahead players
(``lookahead:``, ``value:``) need it.
"""

import json
from pathlib import Path

import numpy as np

from game_types import GameView, SendAction

from ml.encoding.action_encoder import decode_action, legal_actions
from ml.encoding.state_encoder import encode_state, layout_for
from ml.play.numpy_model import NumpyModel, choose
from ml.play.players import NETWORK, NetworkSettings, parse_name


class NumpyAgent:
    def __init__(
        self,
        model: NumpyModel,
        greedy: bool = False,
        seed: int | None = None,
        min_send_gap: int = 0,
        temperature: float = 1.0,
    ):
        """The same settings as ``NeuralAgent``."""

        self.model = model
        self.greedy = greedy
        self.temperature = temperature
        self.min_send_gap = min_send_gap
        self.last_send: dict[int, int] = {}  # by player: the tick of its last send
        self.reseed(seed)

    @classmethod
    def from_init(
        cls,
        init: str | Path,
        greedy: bool = False,
        seed: int | None = None,
        min_send_gap: int = 0,
        temperature: float = 1.0,
    ) -> "NumpyAgent":
        if str(init) == "random":
            raise ValueError("An untrained network needs PyTorch; without it, only a saved checkpoint plays")
        return cls(NumpyModel.from_file(init), greedy, seed, min_send_gap, temperature)

    def reseed(self, seed: int | None) -> None:
        self.generator = np.random.default_rng(seed)

    def choose_action(self, state: GameView) -> SendAction | None:
        last = self.last_send.get(state.player_id)
        if last is not None and state.tick <= last:
            del self.last_send[state.player_id]  # the clock went back: a new game has begun
        elif last is not None and state.tick - last < self.min_send_gap:
            return None

        layout = layout_for(state)
        legal = legal_actions(state, layout)
        if not legal.allowed.any():
            return None
        output = self.model(encode_state(state, layout), legal)
        choice = choose(output, self.greedy, self.generator, self.temperature)
        action = decode_action(choice, legal, free=self.model.free_amounts)
        if action is not None:
            self.last_send[state.player_id] = state.tick
        return action

    __call__ = choose_action


def network_player(name: str, seed: int | None = None) -> NumpyAgent:
    """The player a ``network:PATH?temperature=...`` name stands for, as ``ml.play.players`` builds it."""

    if not name.startswith(NETWORK):
        raise ValueError(f"'{name}' needs PyTorch: without it, only network: players play")
    path, settings = parse_name(name.removeprefix(NETWORK), NetworkSettings)
    return NumpyAgent.from_init(path, seed=seed, temperature=settings.temperature)


def min_send_gap(checkpoint: str | Path) -> int:
    """``ml.training.runlog.min_send_gap``, which cannot be imported without PyTorch."""

    settings = Path(checkpoint).parent / "settings.json"
    if not settings.is_file():
        return 0
    return int(json.loads(settings.read_text()).get("min_send_gap", 0))
