"""The contract between models and everything that uses them.

Every architecture takes a ``ModelInput`` and returns a ``ModelOutput``.
Trainers, the agent, and evaluation only ever touch these two types, so one
architecture can replace another without changes anywhere outside
``ml/models/``.

An architecture only has to produce scores. Removing illegal choices and
squashing the value happen here, identically for every architecture.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from atlas_model.encoding.action_encoder import LegalActions, stack_legal_actions
from atlas_model.encoding.state_encoder import EncodedState, stack_states


# Added to the score of a choice the game would reject. Large enough that its
# probability is exactly zero, finite so the arithmetic never produces NaN.
ILLEGAL = -1e9


@dataclass(frozen=True)
class ModelInput:
    """A batch of positions, padded to the largest map in the batch."""

    base_features: Tensor  # [batch, bases, base features]
    pair_features: Tensor  # [batch, bases, bases, pair features]
    global_features: Tensor  # [batch, global features]
    base_mask: Tensor  # [batch, bases], True for real bases and False for padding
    # [batch, bases, bases, amount options], True where the game accepts the choice.
    # Used to remove illegal choices; an architecture may also read it as an input.
    allowed: Tensor

    def to(self, device: torch.device | str) -> "ModelInput":
        return ModelInput(
            self.base_features.to(device),
            self.pair_features.to(device),
            self.global_features.to(device),
            self.base_mask.to(device),
            self.allowed.to(device),
        )


@dataclass(frozen=True)
class Scores:
    """What an architecture computes. Higher means more preferred; any real number."""

    wait: Tensor  # [batch]
    route: Tensor  # [batch, bases, bases], source then target
    amount: Tensor  # [batch, bases, bases, amount options]
    value: Tensor  # [batch]


@dataclass(frozen=True)
class ModelOutput:
    """Scores with illegal choices removed, ready to choose from.

    The choice has two steps: first wait or a route, then an amount for that
    route. ``ml/models/policy.py`` turns these into choices and probabilities.
    """

    # [batch, 1 + bases * bases]. Column 0 is "wait"; column 1 + source * bases + target is a route.
    route_logits: Tensor
    amount_logits: Tensor  # [batch, bases, bases, amount options]
    value: Tensor  # [batch], expected result from -1 (loss) to +1 (win)


class PolicyValueModel(nn.Module):
    """Base class for every architecture.

    Subclasses implement ``scores`` and keep the config dataclass they were
    built from in ``self.config``, so a saved model can be rebuilt.
    """

    config: Any

    def scores(self, inputs: ModelInput) -> Scores:
        raise NotImplementedError

    def forward(self, inputs: ModelInput) -> ModelOutput:
        scores = self.scores(inputs)
        route_allowed = inputs.allowed.any(dim=-1)
        route = scores.route.masked_fill(~route_allowed, ILLEGAL).flatten(start_dim=1)
        return ModelOutput(
            route_logits=torch.cat((scores.wait[:, None], route), dim=1),
            amount_logits=scores.amount.masked_fill(~inputs.allowed, ILLEGAL),
            value=torch.tanh(scores.value),
        )


def make_input(
    states: list[EncodedState],
    legal: list[LegalActions],
    device: torch.device | str = "cpu",
) -> ModelInput:
    """Stack encoded positions and their legal choices into one batch of tensors."""

    batch = stack_states(states)
    return ModelInput(
        base_features=torch.from_numpy(batch.base_features),
        pair_features=torch.from_numpy(batch.pair_features),
        global_features=torch.from_numpy(batch.global_features),
        base_mask=torch.from_numpy(batch.base_mask),
        allowed=torch.from_numpy(stack_legal_actions(legal)),
    ).to(device)
