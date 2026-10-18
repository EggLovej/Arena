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

import numpy as np

from ml.encoding.action_encoder import LegalActions, SendChoice
from ml.encoding.state_encoder import EncodedState
from ml.play.numpy_model import NumpyModel, choose
from ml.play.players import NETWORK, NetworkAgent, NetworkSettings, parse_name


class NumpyAgent(NetworkAgent):
    @staticmethod
    def load(init, seed: int | None) -> NumpyModel:
        if str(init) == "random":
            raise ValueError("An untrained network needs PyTorch; without it, only a saved checkpoint plays")
        return NumpyModel.from_file(init)

    def reseed(self, seed: int | None) -> None:
        self.generator = np.random.default_rng(seed)

    def decide(self, state: EncodedState, legal: LegalActions) -> SendChoice | None:
        return choose(self.model(state, legal), self.greedy, self.generator, self.temperature)


def network_player(name: str, seed: int | None = None) -> NumpyAgent:
    """The player a ``network:PATH?temperature=...`` name stands for, as ``ml.play.players`` builds it."""

    if not name.startswith(NETWORK):
        raise ValueError(f"'{name}' needs PyTorch: without it, only network: players play")
    path, settings = parse_name(name.removeprefix(NETWORK), NetworkSettings)
    return NumpyAgent.from_init(path, seed=seed, temperature=settings.temperature)
