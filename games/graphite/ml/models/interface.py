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

from ml.encoding.action_encoder import LegalActions, stack_legal_actions, stack_route_limits
from ml.encoding.state_encoder import EncodedState, stack_states


# Added to the score of a choice the game would reject. Large enough that its
# probability is exactly zero, finite so the arithmetic never produces NaN.
ILLEGAL = -1e9


@dataclass(frozen=True)
class ModelInput:
    """A batch of positions, padded to the largest map in the batch."""

    base_features: Tensor  # [batch, bases, base features]
    pair_features: Tensor  # [batch, bases, bases, pair features]
    route_features: Tensor  # [batch, bases, bases, route features]
    global_features: Tensor  # [batch, global features]
    base_mask: Tensor  # [batch, bases], True for real bases and False for padding
    # [batch, bases, bases, amount options], True where the game accepts the choice.
    # Used to remove illegal choices; an architecture may also read it as an input.
    allowed: Tensor
    # For networks with free amounts, per route: the most units it can send and what a capture
    # needs ([batch, bases, bases], unit counts; see LegalActions). None where not provided.
    most: Tensor | None = None
    needed: Tensor | None = None

    def to(self, device: torch.device | str) -> "ModelInput":
        return ModelInput(
            *(None if value is None else value.to(device) for value in (getattr(self, name) for name in _INPUT_FIELDS))
        )


@dataclass(frozen=True)
class Scores:
    """What an architecture computes. Higher means more preferred; any real number."""

    wait: Tensor  # [batch]
    route: Tensor  # [batch, bases, bases], source then target
    # [batch, bases, bases, amount options]. An architecture built before options were added
    # scores fewer; the ones it does not know are never chosen. None with free amounts.
    amount: Tensor | None
    value: Tensor  # [batch]
    # With free amounts, in place of ``amount``: the curves of each route
    # ([batch, bases, bases, curves, parameters]; see ml/models/amounts.py).
    amount_curves: Tensor | None = None


@dataclass(frozen=True)
class ModelOutput:
    """Scores with illegal choices removed, ready to choose from.

    The choice has two steps: first wait or a route, then an amount for that
    route. ``ml/models/policy.py`` turns these into choices and probabilities.
    """

    # [batch, 1 + bases * bases]. Column 0 is "wait"; column 1 + source * bases + target is a route.
    route_logits: Tensor
    amount_logits: Tensor | None  # [batch, bases, bases, amount options]; None with free amounts
    value: Tensor  # [batch], expected result from -1 (loss) to +1 (win)
    # With free amounts: each route's curves, and the two numbers they are built on (as in ModelInput).
    amount_curves: Tensor | None = None
    most: Tensor | None = None
    needed: Tensor | None = None

    def to(self, device: torch.device | str) -> "ModelOutput":
        return ModelOutput(
            *(None if value is None else value.to(device) for value in (getattr(self, name) for name in _OUTPUT_FIELDS))
        )


class PolicyValueModel(nn.Module):
    """Base class for every architecture.

    Subclasses implement ``scores`` and keep the config dataclass they were
    built from in ``self.config``, so a saved model can be rebuilt.
    """

    config: Any

    @property
    def free_amounts(self) -> bool:
        """True for a network that picks any whole number of units instead of an amount option."""

        return getattr(self.config, "amounts", "options") == "free"

    def scores(self, inputs: ModelInput) -> Scores:
        raise NotImplementedError

    def forward(self, inputs: ModelInput) -> ModelOutput:
        scores = self.scores(inputs)
        if scores.amount_curves is not None:
            # Free amounts: a route is usable exactly where an option network may use it, and
            # every amount from 1 to ``most`` is legal, so nothing more is removed here.
            route = scores.route.masked_fill(~inputs.allowed.any(dim=-1), ILLEGAL).flatten(start_dim=1)
            return ModelOutput(
                route_logits=torch.cat((scores.wait[:, None], route), dim=1),
                amount_logits=None,
                value=torch.tanh(scores.value),
                amount_curves=scores.amount_curves,
                most=inputs.most,
                needed=inputs.needed,
            )
        amount, allowed = scores.amount, inputs.allowed
        known, options = amount.shape[-1], allowed.shape[-1]
        if known < options:
            # Options added after this architecture was built are never chosen.
            amount = torch.cat((amount, amount.new_full((*amount.shape[:-1], options - known), ILLEGAL)), dim=-1)
            allowed = allowed.clone()
            allowed[..., known:] = False
        route_allowed = allowed.any(dim=-1)
        route = scores.route.masked_fill(~route_allowed, ILLEGAL).flatten(start_dim=1)
        return ModelOutput(
            route_logits=torch.cat((scores.wait[:, None], route), dim=1),
            amount_logits=amount.masked_fill(~allowed, ILLEGAL),
            value=torch.tanh(scores.value),
        )


def make_input(
    states: list[EncodedState],
    legal: list[LegalActions],
    device: torch.device | str = "cpu",
) -> ModelInput:
    """Stack encoded positions and their legal choices into one batch of tensors."""

    batch = stack_states(states)
    most, needed = stack_route_limits(legal)
    return ModelInput(
        base_features=torch.from_numpy(batch.base_features),
        pair_features=torch.from_numpy(batch.pair_features),
        route_features=torch.from_numpy(batch.route_features),
        global_features=torch.from_numpy(batch.global_features),
        base_mask=torch.from_numpy(batch.base_mask),
        allowed=torch.from_numpy(stack_legal_actions(legal)),
        most=torch.from_numpy(most),
        needed=torch.from_numpy(needed),
    ).to(device)


def padded_length(count: int) -> int:
    """``count`` rounded up to one of a few fixed lengths.

    On the Apple GPU, learning from every new tensor shape costs memory that is
    never given back. Lists whose length changes from batch to batch (the usable
    routes, for instance) are padded to these lengths, so few shapes ever occur.
    Eight lengths per doubling, so padding is always less than a quarter of the
    real length.
    """

    return count + -count % 2 ** max(3, (count - 1).bit_length() - 3)


def padded_route_list(usable: Tensor, batch: int) -> tuple[Tensor, Tensor, Tensor]:
    """Where the usable routes are, as three index lists, padded to a ``padded_length``.

    The padding comes last and points at position ``batch``, one past the real
    ones, where callers add an extra copy that is thrown away afterwards. For
    callers that cannot, see ``readable_route_list``.
    """

    position, source, target = usable.nonzero(as_tuple=True)
    padding = padded_length(len(position)) - len(position)
    if padding:
        position = torch.cat((position, position.new_full((padding,), batch)))
        source = torch.cat((source, source.new_zeros(padding)))
        target = torch.cat((target, target.new_zeros(padding)))
    return position, source, target


def readable_route_list(usable: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """As ``padded_route_list``, but the padding repeats the first route, so every entry can be read.

    The fourth list is True for the real entries; callers leave the rest out of what they add up.
    """

    batch = usable.shape[0]
    position, source, target = padded_route_list(usable, batch)
    real = position < batch
    return (*(torch.where(real, index, index[:1]) for index in (position, source, target)), real)


_INPUT_FIELDS = tuple(ModelInput.__dataclass_fields__)
_OUTPUT_FIELDS = tuple(ModelOutput.__dataclass_fields__)
