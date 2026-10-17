"""Turn a model's output into choices, and measure those choices.

"Policy" is the usual name for the rule an agent uses to pick its move. Here
the rule is a probability for every legal choice, made in two steps:

    P(wait)                              from the route scores
    P(route and amount) = P(route) * P(amount, given that route)

These functions work on any ``ModelOutput``, whichever architecture made it.
A choice is stored as two whole numbers per position: the route column (0 for
wait) and the amount option.
"""

import torch
from torch import Tensor
from torch.nn.functional import log_softmax

from atlas_model.encoding.action_encoder import SendChoice
from atlas_model.models.interface import ModelOutput


def choose(
    output: ModelOutput,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Pick one choice per position: by chance, or the most likely one if ``greedy``.

    Pass a seeded ``generator`` to make the chance picks repeatable.
    """

    route = _pick(output.route_logits, greedy, generator)
    amount = _pick(_amount_logits_for(output, route), greedy, generator)
    return route, torch.where(route == 0, 0, amount)


def log_probability(output: ModelOutput, route: Tensor, amount: Tensor) -> Tensor:
    """How likely the model finds each given choice, as a logarithm. [batch]"""

    route_part = log_softmax(output.route_logits, dim=1).gather(1, route[:, None])[:, 0]
    amount_logits = _amount_logits_for(output, route)
    amount_part = log_softmax(amount_logits, dim=1).gather(1, amount[:, None])[:, 0]
    # Waiting has no amount, so only the first step counts.
    return route_part + torch.where(route == 0, 0.0, amount_part)


def entropy(output: ModelOutput) -> Tensor:
    """How undecided the model is: 0 when certain, larger the more evenly it spreads. [batch]"""

    route_log = log_softmax(output.route_logits, dim=1)
    route_probability = route_log.exp()
    route_entropy = -(route_probability * route_log).sum(dim=1)

    amount_log = log_softmax(output.amount_logits, dim=-1)
    amount_entropy = -(amount_log.exp() * amount_log).sum(dim=-1).flatten(start_dim=1)
    # The amount only matters as often as its route is chosen.
    return route_entropy + (route_probability[:, 1:] * amount_entropy).sum(dim=1)


def to_choices(route: Tensor, amount: Tensor, bases: int) -> list[SendChoice | None]:
    """Choices as the action encoder expects them. ``bases`` is the batch's padded map size."""

    choices: list[SendChoice | None] = []
    for column, option in zip(route.tolist(), amount.tolist()):
        if column == 0:
            choices.append(None)
        else:
            source, target = divmod(column - 1, bases)
            choices.append(SendChoice(source, target, option))
    return choices


def from_choices(
    choices: list[SendChoice | None],
    bases: int,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """The reverse of ``to_choices``, for scoring choices that were made earlier."""

    route = [0 if choice is None else 1 + choice.source * bases + choice.target for choice in choices]
    amount = [0 if choice is None else choice.amount for choice in choices]
    return torch.tensor(route, device=device), torch.tensor(amount, device=device)


def _amount_logits_for(output: ModelOutput, route: Tensor) -> Tensor:
    """The amount scores of each position's chosen route. [batch, amount options]"""

    per_route = output.amount_logits.flatten(start_dim=1, end_dim=2)
    # "Wait" borrows route 0's scores; callers ignore the result in that case.
    pair = (route - 1).clamp(min=0)
    return per_route[torch.arange(len(route), device=route.device), pair]


def _pick(logits: Tensor, greedy: bool, generator: torch.Generator | None) -> Tensor:
    if greedy:
        return logits.argmax(dim=1)
    probability = torch.softmax(logits, dim=1)
    return torch.multinomial(probability, num_samples=1, generator=generator)[:, 0]
