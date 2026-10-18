"""Ratings from match results, for any number of players per match.

Each match is a ranking of its players (ties allowed). The ratings are the
Plackett-Luce model fitted to all results at once: every player has a strength,
and the chance that a given player finishes first among some set of players is
its strength divided by the set's total strength. With two players this is
exactly the Elo model, so ratings are shown on the Elo scale: 400 points more
means winning 10 times as often, 100 points more about 64% of games.

Fitting all results at once (rather than updating after every match, as Elo or
OpenSkill do) means the order of matches doesn't matter and rerunning a
tournament from scratch gives the same ratings for the same results.

A tie is counted as every order of the tied players, equally likely, so a drawn
two-player match is half a win for each. Every player also gets one imaginary
win and one imaginary loss against an average opponent, which keeps a player
that has won (or lost) every match at a finite rating.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import permutations, product


SCALE = 400 / math.log(10)  # Elo points per unit of log-strength
PRIOR_GAMES = 1.0  # imaginary wins and losses each player gets against an average opponent
_AVERAGE = object()  # the imaginary opponent; its strength stays 1

Ranking = Sequence[Sequence[str]]  # groups of players, best first; a group shares a place


@dataclass(frozen=True)
class Rating:
    rating: float
    interval: float  # 95% of the time, the true rating (relative to the others) is within this many points


def fit(rankings: Iterable[Ranking], anchor: tuple[str, float] | None = None, tolerance: float = 1e-10) -> dict[str, Rating]:
    """Ratings for every player in ``rankings``.

    With ``anchor`` (player, rating), that player gets exactly that rating; otherwise
    the average rating is 1500.
    """

    stages = _stages(rankings)  # (chosen, choice set, weight)
    players = sorted({player for _, choice, _ in stages for player in choice if player is not _AVERAGE})
    if not players:
        return {}
    for player in players:
        stages.append((player, (player, _AVERAGE), PRIOR_GAMES))
        stages.append((_AVERAGE, (player, _AVERAGE), PRIOR_GAMES))

    # Minorize-maximize (Hunter, 2004): each step can only improve the fit.
    strength = {player: 1.0 for player in players}
    strength[_AVERAGE] = 1.0
    wins = {player: 0.0 for player in players}
    for chosen, _, weight in stages:
        if chosen is not _AVERAGE:
            wins[chosen] += weight
    for _ in range(100_000):
        denominator = {player: 0.0 for player in players}
        for _, choice, weight in stages:
            share = weight / sum(strength[player] for player in choice)
            for player in choice:
                if player is not _AVERAGE:
                    denominator[player] += share
        updated = {player: wins[player] / denominator[player] for player in players}
        change = max(abs(math.log(updated[player] / strength[player])) for player in players)
        strength.update(updated)
        if change < tolerance:
            break

    log_strength = {player: math.log(strength[player]) for player in players}
    variance = _variances(players, stages, strength)
    if anchor is not None and anchor[0] in log_strength:
        offset = anchor[1] - SCALE * log_strength[anchor[0]]
    else:
        offset = 1500 - SCALE * sum(log_strength.values()) / len(players)
    return {
        player: Rating(offset + SCALE * log_strength[player], 1.96 * SCALE * math.sqrt(variance[player]))
        for player in players
    }


def _orders(ranking: Ranking) -> list[tuple[tuple[str, ...], float]]:
    """Every strict order consistent with a ranking with ties, and how much each counts."""

    options = []
    for group in ranking:
        group = tuple(group)
        if len(group) <= 4:
            options.append(list(permutations(group)))
        else:
            # Too many orders to list: each player leading the group once is a fair stand-in.
            options.append([group[index:] + group[:index] for index in range(len(group))])
    combinations = list(product(*options))
    weight = 1 / len(combinations)
    return [(tuple(player for group in combination for player in group), weight) for combination in combinations]


def _stages(rankings: Iterable[Ranking]) -> list[tuple[object, tuple, float]]:
    stages = []
    for ranking in rankings:
        for order, weight in _orders(ranking):
            for index in range(len(order) - 1):
                stages.append((order[index], order[index:], weight))
    return stages


def _variances(players: list[str], stages, strength) -> dict[str, float]:
    """The uncertainty of each log-strength compared with the average of all players.

    Ratings only mean something relative to each other (the anchor just shifts them all),
    so the uncertainty of the overall level, which only the imaginary games pin down, is
    left out: the variance of a player's log-strength minus the mean log-strength.
    """

    index = {player: number for number, player in enumerate(players)}
    size = len(players)
    curvature = [[0.0] * size for _ in range(size)]
    for _, choice, weight in stages:
        total = sum(strength[player] for player in choice)
        shares = [(index[player], strength[player] / total) for player in choice if player is not _AVERAGE]
        for a, share_a in shares:
            curvature[a][a] += weight * share_a
            for b, share_b in shares:
                curvature[a][b] -= weight * share_a * share_b
    inverse = _invert(curvature)
    row_means = [sum(row) / size for row in inverse]
    overall = sum(row_means) / size
    return {
        player: max(inverse[number][number] - 2 * row_means[number] + overall, 0.0)
        for player, number in index.items()
    }


def _invert(matrix: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan elimination; the matrices here are small and well conditioned."""

    size = len(matrix)
    work = [row[:] + [float(column == number) for column in range(size)] for number, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        work[column], work[pivot] = work[pivot], work[column]
        lead = work[column][column]
        if abs(lead) < 1e-300:
            raise ValueError("The results do not determine the ratings")
        work[column] = [value / lead for value in work[column]]
        for row in range(size):
            if row != column and work[row][column]:
                factor = work[row][column]
                work[row] = [value - factor * pivot_value for value, pivot_value in zip(work[row], work[column])]
    return [row[size:] for row in work]
