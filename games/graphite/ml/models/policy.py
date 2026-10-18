"""Turn a model's output into choices, and measure those choices.

"Policy" is the usual name for the rule an agent uses to pick its move. Here
the rule is a probability for every legal choice, made in two steps:

    P(wait)                              from the route scores
    P(route and amount) = P(route) * P(amount, given that route)

These functions work on any ``ModelOutput``, whichever architecture made it.
A choice is stored as two whole numbers per position: the route column (0 for
wait) and the amount option, or, for a network with free amounts, the units
(the amount maths is in ``ml/models/amounts.py``).
"""

import torch
from torch import Tensor
from torch.nn.functional import log_softmax

from ml.encoding.action_encoder import SendChoice
from ml.models import amounts
from ml.models.interface import ModelOutput, readable_route_list


def choose(
    output: ModelOutput,
    greedy: bool = False,
    generator: torch.Generator | None = None,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Pick one choice per position: by chance, or the most likely one if ``greedy``.

    ``temperature`` below 1 draws the likelier choices more often still: the route's
    odds, and then the amount's, are raised to the power 1 / temperature and scaled to
    add up to 1 again (at 0.5, odds of 60/30/10% become 78/20/2%). 1 draws by the
    network's own odds, as in training. Pass a seeded ``generator`` to make the chance
    picks repeatable.
    """

    if temperature <= 0:
        raise ValueError(f"temperature must be above 0, not {temperature}")
    sharpen = 1.0 / temperature
    route = _pick(output.route_logits * sharpen, greedy, generator)
    if output.amount_curves is not None:
        curves, most, needed = _free_amounts_for(output, route)
        if greedy:
            amount = amounts.most_likely(curves, most, needed)
        elif sharpen != 1.0:
            amount = amounts.sample_sharpened(curves, most, needed, sharpen, generator)
        else:
            amount = amounts.sample(curves, most, needed, generator)
    else:
        amount = _pick(_amount_logits_for(output, route) * sharpen, greedy, generator)
    return route, torch.where(route == 0, 0, amount)


def log_probability(output: ModelOutput, route: Tensor, amount: Tensor) -> Tensor:
    """How likely the model finds each given choice, as a logarithm. [batch]"""

    route_part = log_softmax(output.route_logits, dim=1).gather(1, route[:, None])[:, 0]
    if output.amount_curves is not None:
        curves, most, needed = _free_amounts_for(output, route)
        amount_part = amounts.log_probability(curves, most, needed, amount.clamp(min=1))
    else:
        amount_logits = _amount_logits_for(output, route)
        amount_part = log_softmax(amount_logits, dim=1).gather(1, amount[:, None])[:, 0]
    # Waiting has no amount, so only the first step counts.
    return route_part + torch.where(route == 0, 0.0, amount_part)


def entropy(output: ModelOutput) -> Tensor:
    """How undecided the model is: 0 when certain, larger the more evenly it spreads. [batch]"""

    route_part, amount_part = entropy_parts(output)
    return route_part + amount_part


def entropy_parts(output: ModelOutput) -> tuple[Tensor, Tensor]:
    """``entropy`` in its two parts, which add up to it: [batch] each

    - how undecided the route is (waiting included);
    - how undecided the amount is once the route is chosen, averaged over the
      routes by how likely each is (waiting has no amount and adds nothing).
    """

    route_log = log_softmax(output.route_logits, dim=1)
    route_probability = route_log.exp()
    route_entropy = -(route_probability * route_log).sum(dim=1)

    if output.amount_curves is not None:
        # Only for the usable routes: a table over every pair of bases and every amount would be huge.
        # The list is padded for the Apple GPU; the padding counts for nothing.
        position, source, target, real = readable_route_list(output.most > 0)
        bases = output.most.shape[1]
        per_route = amounts.entropy(
            output.amount_curves[position, source, target],
            output.most[position, source, target],
            output.needed[position, source, target],
        )
        weight = route_probability[position, 1 + source * bases + target] * real
        return route_entropy, torch.zeros_like(route_entropy).index_add(0, position, weight * per_route)

    amount_log = log_softmax(output.amount_logits, dim=-1)
    amount_entropy = -(amount_log.exp() * amount_log).sum(dim=-1).flatten(start_dim=1)
    # The amount only matters as often as its route is chosen.
    return route_entropy, (route_probability[:, 1:] * amount_entropy).sum(dim=1)


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


def _free_amounts_for(output: ModelOutput, route: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The curves, ``most`` and ``needed`` of each position's chosen route ("wait" borrows route 0's)."""

    pair = (route - 1).clamp(min=0)
    rows = torch.arange(len(route), device=route.device)
    curves = output.amount_curves.flatten(start_dim=1, end_dim=2)[rows, pair]
    most = output.most.flatten(start_dim=1)[rows, pair]
    needed = output.needed.flatten(start_dim=1)[rows, pair]
    return curves, most, needed


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
