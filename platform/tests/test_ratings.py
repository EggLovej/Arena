import math
import random

import pytest

from arena.ratings import expected_score, fit


def games(winner, loser, count):
    return [[[winner], [loser]]] * count


def test_two_players_follow_elo():
    # 3 wins in 4 is a 190.8 point gap on the Elo scale; the imaginary games pull it in a little.
    ratings = fit(games("a", "b", 300) + games("b", "a", 100))
    gap = ratings["a"].rating - ratings["b"].rating
    assert gap == pytest.approx(400 * math.log10(3), abs=2)
    assert expected_score(ratings["a"].rating, ratings["b"].rating) == pytest.approx(0.75, abs=0.003)


def test_draws_count_half_each_way():
    drawn = fit([[["a", "b"]]] * 10)
    assert drawn["a"].rating == pytest.approx(drawn["b"].rating)
    mixed = fit(games("a", "b", 5) + [[["a", "b"]]] * 10)
    assert fit(games("a", "b", 10) + games("b", "a", 5))["a"].rating == pytest.approx(mixed["a"].rating, abs=1e-6)


def test_average_is_1500_and_the_anchor_is_exact():
    results = games("a", "b", 6) + games("b", "c", 6) + games("a", "c", 3) + games("c", "a", 1)
    free = fit(results)
    assert sum(rating.rating for rating in free.values()) / 3 == pytest.approx(1500)
    anchored = fit(results, anchor=("b", 1000))
    assert anchored["b"].rating == pytest.approx(1000)
    for player in "abc":  # the anchor shifts every rating by the same amount
        assert anchored[player].rating - free[player].rating == pytest.approx(1000 - free["b"].rating)
        assert anchored[player].interval == pytest.approx(free[player].interval)


def test_an_unbeaten_player_gets_a_finite_rating():
    ratings = fit(games("a", "b", 20))
    assert math.isfinite(ratings["a"].rating)
    assert ratings["a"].rating > ratings["b"].rating + 300


def test_order_of_matches_does_not_matter():
    rng = random.Random(4)
    players = "abcd"
    results = []
    for _ in range(60):
        order = rng.sample(players, 3)
        results.append([[order[0]], [order[1], order[2]]] if rng.random() < 0.3 else [[player] for player in order])
    shuffled = results[:]
    rng.shuffle(shuffled)
    first, second = fit(results), fit(shuffled)
    for player in players:
        assert first[player].rating == pytest.approx(second[player].rating, abs=1e-6)


def test_multiplayer_rankings_order_the_players():
    ratings = fit([[["a"], ["b"], ["c"]]] * 10 + [[["b"], ["a"], ["c"]]] * 3)
    assert ratings["a"].rating > ratings["b"].rating > ratings["c"].rating


def test_more_games_mean_a_narrower_interval():
    few = fit(games("a", "b", 3) + games("b", "a", 3))
    many = fit(games("a", "b", 300) + games("b", "a", 300))
    assert many["a"].interval < few["a"].interval / 5
    # 600 even games: the gap is known to about +-1.96 * 173.7 / sqrt(150) = +-28, each side half of it.
    assert many["a"].interval == pytest.approx(14, abs=1)


def test_no_results_no_ratings():
    assert fit([]) == {}
