"""Let a model play the game: GameView in, SendAction out.

This is the glue between the four earlier pieces. Each decision:

    GameView -> state encoder  -> what the network sees
             -> action encoder -> which choices are legal
             -> model          -> a score for every choice
             -> policy         -> one choice
             -> action encoder -> SendAction

One position at a time on the CPU, which is what playing in the game window
needs. Training plays many games at once and batches this instead.

A player called ``network:PATH`` is this agent wherever players are named
(the evaluator, the rollouts, Beast), for example
``network:ml/runs/free_gap0/round_0050.pt?temperature=0.2``.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from game_types import GameView, SendAction

from ml.encoding.action_encoder import decode_action, legal_actions
from ml.encoding.state_encoder import encode_state, layout_for
from ml.models import PolicyValueModel, make_input
from ml.models.checkpoint import init_model
from ml.models.policy import choose, to_choices


PREFIX = "network:"


@dataclass(frozen=True)
class NetworkSettings:
    temperature: float = 1.0  # 1 draws by the network's own odds; Neural AI plays at 0.2

    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError(f"temperature must be above 0, not {self.temperature}")


class NeuralAgent:
    def __init__(
        self,
        model: PolicyValueModel,
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

        self.model = model.to("cpu").eval()
        self.greedy = greedy
        self.temperature = temperature
        self.min_send_gap = min_send_gap
        self.last_send: dict[int, int] = {}  # by player: the tick of its last send
        self.generator = torch.Generator()
        self.reseed(seed)

    @classmethod
    def from_init(
        cls,
        init: str | Path = "random",
        greedy: bool = False,
        seed: int | None = None,
        min_send_gap: int = 0,
        temperature: float = 1.0,
    ) -> "NeuralAgent":
        """Build from "random" or a checkpoint path, as trainers do."""

        return cls(init_model(init, seed=seed).model, greedy, seed, min_send_gap, temperature)

    def reseed(self, seed: int | None) -> None:
        """Make the chance picks repeatable from here on; ``None`` picks a fresh seed."""

        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)

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

        inputs = make_input([encode_state(state, layout)], [legal])
        with torch.no_grad():
            output = self.model(inputs)
        route, amount = choose(output, self.greedy, self.generator, self.temperature)
        choice = to_choices(route, amount, bases=len(legal.base_ids))[0]
        action = decode_action(choice, legal, free=self.model.free_amounts)
        if action is not None:
            self.last_send[state.player_id] = state.tick
        return action

    __call__ = choose_action
