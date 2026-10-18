"""Let a model play the game with PyTorch: GameView in, SendAction out.

One position at a time on the CPU, which is what playing in the game window
needs. Training plays many games at once and batches this instead. Every step
but working out the network is ``NetworkAgent``'s (ml/play/players.py).

A player called ``network:PATH`` is this agent wherever players are named
(the evaluator, the rollouts, Beast), for example
``network:ml/runs/free_gap0/round_0050.pt?temperature=0.2``.
"""

import torch

from ml.encoding.action_encoder import LegalActions, SendChoice
from ml.encoding.state_encoder import EncodedState
from ml.models import PolicyValueModel, make_input
from ml.models.checkpoint import init_model
from ml.models.policy import choose, to_choices
from ml.play.players import NETWORK as PREFIX, NetworkAgent, NetworkSettings  # noqa: F401 (also read from here)


class NeuralAgent(NetworkAgent):
    def __init__(self, model: PolicyValueModel, *settings, **named_settings):
        """The settings of ``NetworkAgent``."""

        super().__init__(model.to("cpu").eval(), *settings, **named_settings)

    @staticmethod
    def load(init, seed: int | None) -> PolicyValueModel:
        return init_model(init, seed=seed).model

    def reseed(self, seed: int | None) -> None:
        """Make the chance picks repeatable from here on; ``None`` picks a fresh seed."""

        self.generator = torch.Generator()
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)

    def decide(self, state: EncodedState, legal: LegalActions) -> SendChoice | None:
        with torch.no_grad():
            output = self.model(make_input([state], [legal]))
        route, amount = choose(output, self.greedy, self.generator, self.temperature)
        return to_choices(route, amount, bases=len(legal.base_ids))[0]
