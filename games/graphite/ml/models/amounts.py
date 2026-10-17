"""Free amounts: any whole number of units, from a blend of a few curves per route.

An option network picks one of nine amounts ("just enough" and eighths of the
units). A network with free amounts may send any whole number of units, from
1 to everything at the source (``most``). For each route it describes where
it wants to land with a few curves (three by default), each with:

- a weight: how much of the probability it carries;
- a centre, built from two numbers the game provides, so that exact amounts
  take no arithmetic inside the network:

      centre = fraction * most + share * needed + offset

  ``needed`` is the smallest fleet that captures the target (0 when it will
  be mine anyway). Fraction 1 alone is "everything", share 1 alone is "just
  enough", and an offset of 2 adds two spare units;
- a width, in units: narrow for "exactly this", wide for "about this".

Each curve is a logistic, a bell shape whose area up to x is simply
sigmoid((x - centre) / width). The probability of sending k units is the
blend's area between k - 1/2 and k + 1/2, except that everything below 1.5
belongs to 1 and everything above most - 1/2 to most. So the probabilities of
1, 2, ..., most add up to exactly one, and every amount is one choice: no two
choices send the same units.

The network gives five numbers per curve (``PARAMETERS``); ``curves`` turns
them into weights, centres and widths.
"""

import torch
from torch import Tensor
from torch.nn.functional import log_softmax, logsigmoid, softplus

from ml.models.interface import padded_length


CURVES = 3  # the default number of curves per route
PARAMETERS = 5  # per curve: weight, fraction, share, offset, width
MIN_WIDTH = 0.05  # units; at this width 99.99% of a curve falls on one whole number
# Above this many units, entropy and the distribution tables lump the rest into the last cell.
GRID_LIMIT = 256
# The grids entropy is worked out on: each route on the smallest that holds its most.
GRID_SIZES = (16, 32, 64, 128, GRID_LIMIT)


def curves(parameters: Tensor, most: Tensor, needed: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Log-weights, centres and widths of the curves.

    ``parameters`` is [..., curves, PARAMETERS]; ``most`` and ``needed`` are [...].
    """

    log_weight = log_softmax(parameters[..., 0], dim=-1)
    fraction = torch.sigmoid(parameters[..., 1])  # 0 to 1 of the units at the source
    share = 2 * torch.sigmoid(parameters[..., 2])  # 0 to 2 times what a capture needs
    centre = fraction * most[..., None] + share * needed[..., None] + parameters[..., 3]
    width = softplus(parameters[..., 4]) + MIN_WIDTH
    return log_weight, centre, width


def log_probability(parameters: Tensor, most: Tensor, needed: Tensor, units: Tensor) -> Tensor:
    """The log-probability of sending exactly ``units``, for each route given ([...] each)."""

    log_weight, centre, width = curves(parameters, most, needed)
    per_curve = _log_area(units[..., None].to(centre.dtype), centre, width, most[..., None])
    return torch.logsumexp(log_weight + per_curve, dim=-1)


def on_grid(parameters: Tensor, most: Tensor, needed: Tensor, size: int) -> Tensor:
    """Probabilities of sending 1, 2, ..., ``size`` units, for each route: [..., size].

    Amounts above a route's ``most`` get 0. Where ``most`` is larger than ``size``,
    the last cell holds everything from ``size`` up.
    """

    units = torch.arange(1, size + 1, device=parameters.device, dtype=parameters.dtype)
    units = units.expand(*most.shape, size)
    log_weight, centre, width = curves(parameters, most, needed)
    capped = most.clamp(max=size)[..., None, None]
    per_curve = _log_area(units[..., None], centre[..., None, :], width[..., None, :], capped)
    probability = torch.logsumexp(log_weight[..., None, :] + per_curve, dim=-1).exp()
    return probability * (units <= most[..., None])


def entropy(parameters: Tensor, most: Tensor, needed: Tensor) -> Tensor:
    """How undecided each route's amount is, in the same units as the route choice's entropy.

    ``most`` and ``needed`` are [routes]. Most routes have only a few units, so each
    group of routes is worked out on the smallest grid of GRID_SIZES that holds it,
    and each group is padded to a ``padded_length``: learning on the Apple GPU then
    sees only a few tensor shapes (see ``padded_length``).
    """

    result = torch.zeros_like(most, dtype=parameters.dtype)
    group = torch.bucketize(most.clamp(max=GRID_LIMIT), most.new_tensor(GRID_SIZES))
    for number, size in enumerate(GRID_SIZES):
        index = (group == number).nonzero()[:, 0]
        count = len(index)
        if not count:
            continue
        # The padding repeats the group's first route and counts for nothing.
        index = torch.cat((index, index[:1].expand(padded_length(count) - count)))
        real = torch.arange(len(index), device=index.device) < count
        probability = on_grid(parameters[index], most[index], needed[index], size)
        per_route = -(probability * torch.log(probability.clamp(min=1e-30))).sum(dim=-1)
        result = result.index_add(0, index, per_route * real)
    return result


def sample(parameters: Tensor, most: Tensor, needed: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """One amount per route, drawn by chance: first a curve by its weight, then a point on it."""

    log_weight, centre, width = curves(parameters, most, needed)
    curve = torch.multinomial(log_weight.exp(), num_samples=1, generator=generator)
    centre, width = centre.gather(-1, curve)[..., 0], width.gather(-1, curve)[..., 0]
    uniform = torch.rand(centre.shape, generator=generator, device=centre.device).clamp(1e-7, 1 - 1e-7)
    point = centre + width * (torch.log(uniform) - torch.log1p(-uniform))  # a logistic draw
    return torch.minimum(point.round().clamp(min=1), most.clamp(min=1)).long()


def sample_sharpened(
    parameters: Tensor, most: Tensor, needed: Tensor, power: float, generator: torch.Generator | None = None
) -> Tensor:
    """One amount per route, drawn by chance with every amount's odds raised to ``power``.

    Above 1, the likelier amounts are drawn more often still (see ``policy.choose``).
    """

    size = int(max(most.max().item(), 1))
    probability = on_grid(parameters, most, needed, size) ** power
    # A route with nothing to send (where the choice was to wait) still needs something to
    # draw from; its amount is not used.
    probability[..., 0] += (probability.sum(dim=-1) == 0).to(probability.dtype)
    drawn = torch.multinomial(probability.reshape(-1, size), num_samples=1, generator=generator)
    return drawn.reshape(most.shape) + 1


def most_likely(parameters: Tensor, most: Tensor, needed: Tensor) -> Tensor:
    """The single most likely amount per route, for greedy play."""

    size = int(max(most.max().item(), 1))
    return on_grid(parameters, most, needed, size).argmax(dim=-1) + 1


def starting_biases(curves: int = CURVES) -> Tensor:
    """Where an untrained network's curves begin: "just enough", "half", and "everything".

    These three are the most used of the old options. Further curves start at "half".
    """

    width = -0.7  # softplus(-0.7) + MIN_WIDTH: just under half a unit
    places = [
        (0.0, -6.0, 0.0, 0.0, width),  # fraction 0, share 1: just enough
        (0.0, 0.0, -6.0, 0.0, width),  # fraction 1/2, share 0: half
        (0.0, 6.0, -6.0, 1.0, width),  # fraction 1, share 0, one more: everything
    ]
    places += [places[1]] * max(0, curves - len(places))
    return torch.tensor(places[:curves])


def _log_area(units: Tensor, centre: Tensor, width: Tensor, most: Tensor) -> Tensor:
    """The log of one curve's area over a whole number of units, the edges taking the tails."""

    upper = (units + 0.5 - centre) / width
    lower = (units - 0.5 - centre) / width
    area = _log_sigmoid_difference(upper, lower)
    area = torch.where(units <= 1, logsigmoid(upper), area)  # everything below 1.5 is 1
    area = torch.where(units >= most, logsigmoid(-lower), area)  # everything above most - 1/2 is most
    return torch.where((units <= 1) & (units >= most), torch.zeros_like(area), area)


def _log_sigmoid_difference(high: Tensor, low: Tensor) -> Tensor:
    """log(sigmoid(high) - sigmoid(low)) for high > low, precise far out in either tail.

    Far to the right both sigmoids are nearly 1, and their difference would be lost to
    rounding; there the same difference is taken on the mirror side, where both are tiny.
    """

    mirror = low > 0
    high, low = torch.where(mirror, -low, high), torch.where(mirror, -high, low)
    gap = (logsigmoid(low) - logsigmoid(high)).clamp(max=-1e-30)  # below 0; kept off 0 for the gradient
    # log(1 - exp(gap)), each way where it is precise.
    return logsigmoid(high) + torch.where(gap > -0.693, torch.log(-torch.expm1(gap)), torch.log1p(-torch.exp(gap)))
