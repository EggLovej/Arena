"""Players by name.

Wherever a player is named (the evaluator, the rollouts and so every
trainer, Beast), ``load_player`` turns the name into a player: an AI file in
``ais/``, a checkpoint path, ``untrained`` (``untrained:7`` picks the seed of
its random weights), or a checkpoint at a chosen temperature,
``network:PATH?temperature=0.2``.

A network plays through ``NetworkAgent``: with PyTorch as
``NeuralAgent`` (ml/play/agent.py), without it as ``NumpyAgent``
(ml/play/numpy_agent.py).
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

from game_types import GameView, SendAction
from loading import load_ai

from ml.encoding.action_encoder import decode_action, legal_actions
from ml.encoding.state_encoder import encode_state, layout_for
from ml.play.numpy_model import read_checkpoint


NETWORK = "network:"
GAME_FOLDER = Path(__file__).resolve().parents[2]
UNTRAINED = "untrained"

Controller = Callable[[GameView], SendAction | None]


@dataclass(frozen=True)
class NetworkSettings:
    """The settings of a ``network:`` player. Kept here, away from PyTorch, so that playing
    without it (ml/play/numpy_agent.py) reads the same names."""

    temperature: float = 1.0  # 1 draws by the network's own odds; Neural AI plays at 0.2

    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError(f"temperature must be above 0, not {self.temperature}")


class NetworkAgent:
    """A network playing on its own: GameView in, SendAction out.

    Each decision: the state encoder gives what the network sees, the action
    encoder which choices are legal, the network a score for every choice, the
    policy one choice, and the action encoder turns it into a SendAction.
    ``NeuralAgent`` and ``NumpyAgent`` only differ in how they read a network
    (``load``), work it out (``decide``) and draw random numbers (``reseed``).
    """

    def __init__(
        self,
        model,
        greedy: bool = False,
        seed: int | None = None,
        min_send_gap: int = 0,
        temperature: float = 1.0,
    ):
        """``greedy`` always plays the most likely choice; otherwise choices are drawn by chance,
        at ``temperature`` (below 1: the likelier choices more often still; see ``choose``).

        With a ``min_send_gap``, each player this agent plays for waits that many ticks after
        each of its sends (10: one send a second), as in training with ``--min-send-gap``.
        """

        self.model = model
        self.greedy = greedy
        self.temperature = temperature
        self.min_send_gap = min_send_gap
        self.last_send: dict[int, int] = {}  # by player: the tick of its last send
        self.reseed(seed)

    @classmethod
    def from_init(
        cls,
        init: str | Path = "random",
        greedy: bool = False,
        seed: int | None = None,
        min_send_gap: int = 0,
        temperature: float = 1.0,
    ):
        """Build from "random" or a checkpoint path, as trainers do."""

        return cls(cls.load(init, seed), greedy, seed, min_send_gap, temperature)

    def choose_action(self, state: GameView) -> SendAction | None:
        last = self.last_send.get(state.player_id)
        if last is not None and state.tick <= last:
            del self.last_send[state.player_id]  # the clock went back: a new game has begun
        elif last is not None and state.tick - last < self.min_send_gap:
            return None

        layout = layout_for(state)
        legal = legal_actions(state, layout)
        if not legal.allowed.any():
            # Waiting is the only option, so there is nothing to ask the model.
            return None
        choice = self.decide(encode_state(state, layout), legal)
        action = decode_action(choice, legal, free=self.model.free_amounts)
        if action is not None:
            self.last_send[state.player_id] = state.tick
        return action

    __call__ = choose_action


def load_player(name: str, greedy: bool = False, seed: int | None = None) -> Controller:
    """Turn a player's name into something that chooses actions.

    ``greedy`` and ``seed`` are for the networks that draw their moves by chance:
    checkpoints, ``untrained`` and ``network:`` players.
    """

    if name.startswith(NETWORK):
        from ml.play.agent import NeuralAgent

        return network_player(name, NeuralAgent, seed, greedy)
    if name == UNTRAINED or name.startswith(UNTRAINED + ":") or name.endswith(".pt"):
        from ml.play.agent import NeuralAgent

        if name not in _models:
            from ml.models.checkpoint import init_model

            if name.endswith(".pt"):
                _models[name] = init_model(game_path(name)).model
            else:
                weights_seed = int(name.partition(":")[2] or 0)
                _models[name] = init_model("random", seed=weights_seed).model
        return NeuralAgent(_models[name], greedy, seed)
    return fresh_ai(name)


# Models already loaded in this process, so each worker reads a checkpoint only once.
_models: dict = {}


def keeps_memory(name: str) -> bool:
    """Whether an AI in ais/ remembers things during a match, and so needs its own player per match.

    Such an AI defines ``new_player()``, which returns a player with fresh
    memory. Its plain ``choose_action`` is meant for one match at a time.
    """

    if name == UNTRAINED or not name.removesuffix(".py").isidentifier():
        return False
    choose_action = load_ai(name)
    return callable(getattr(sys.modules[choose_action.__module__], "new_player", None))


def fresh_ai(name: str) -> Controller:
    """An AI from ais/, with fresh memory if it keeps any (see ``keeps_memory``)."""

    choose_action = load_ai(name)
    if keeps_memory(name):
        return sys.modules[choose_action.__module__].new_player()
    return choose_action


def network_player(name: str, agent: type[NetworkAgent], seed: int | None = None, greedy: bool = False):
    """The player a ``network:PATH?key=value&...`` name stands for, played by ``agent``
    (``NeuralAgent`` or ``NumpyAgent``): the path (see ``game_path``) with its ``NetworkSettings``,
    each value read as its setting's type."""

    if not name.startswith(NETWORK):
        raise ValueError(f"'{name}' is not a network: player")
    path, _, options = name.removeprefix(NETWORK).partition("?")
    defaults = {setting.name: setting.default for setting in fields(NetworkSettings)}
    changes = {}
    for option in filter(None, options.split("&")):
        key, _, value = option.partition("=")
        if key not in defaults:
            raise ValueError(f"'{key}' is not a setting here. Settings: {', '.join(defaults)}")
        changes[key] = type(defaults[key])(value)
    settings = NetworkSettings(**changes)
    return agent.from_init(game_path(path), greedy, seed, temperature=settings.temperature)


def game_path(path: str) -> str:
    """A checkpoint named in a player's name: a relative path is read from the game folder,
    wherever the program runs, and "random" stays untrained weights."""

    return path if path == "random" else str(GAME_FOLDER / path)


def min_send_gap(checkpoint: str | Path) -> int:
    """The send limit (``--min-send-gap``) a network was trained under, from its own history; 0 for none.

    Every self-play checkpoint, round 0 included, says so in its last history
    entry, so a copy plays under the same limit wherever it goes. Measurements
    and Neural AI play a network under its limit. Read without PyTorch.
    """

    history = read_checkpoint(checkpoint)["history"]
    return history[-1].get("min_send_gap", 0) if history else 0

