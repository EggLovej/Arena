import unittest

import torch

from ml.models import amounts
from ml.models.amounts import (
    GRID_LIMIT, GRID_SIZES, PARAMETERS, entropy, log_probability, on_grid, sample, sample_sharpened,
)


def narrow(fraction, share, offset):
    """One curve, almost no width, centred on fraction * most + share * needed + offset."""

    logit = lambda p: torch.logit(torch.tensor(p).clamp(1e-6, 1 - 1e-6)).item()
    return torch.tensor([[[0.0, logit(fraction), logit(share / 2), offset, -20.0]]])


class AmountTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        self.most = torch.tensor([1.0, 2.0, 3.0, 7.0, 20.0, 150.0, 300.0])
        self.needed = torch.tensor([0.0, 1.0, 5.0, 3.0, 12.0, 0.0, 900.0])
        self.curves = torch.randn(len(self.most), 3, PARAMETERS, generator=generator) * 2

    def test_probabilities_of_every_amount_add_up_to_one(self):
        table = on_grid(self.curves, self.most, self.needed, 300)
        self.assertTrue(torch.allclose(table.sum(dim=1), torch.ones(len(self.most)), atol=1e-5))
        # Nothing above what the source has.
        for row, most in enumerate(self.most.long().tolist()):
            self.assertEqual(table[row, most:].sum().item(), 0.0)
        # A base larger than the grid lumps the rest into the last cell.
        table = on_grid(self.curves[:1], torch.tensor([1000.0]), self.needed[:1], GRID_LIMIT)
        self.assertAlmostEqual(table.sum().item(), 1.0, places=5)
        self.assertTrue(torch.isfinite(entropy(self.curves[:1], torch.tensor([1000.0]), self.needed[:1])).all())

    def test_log_probability_agrees_with_the_table(self):
        table = on_grid(self.curves, self.most, self.needed, 300)
        for units in (1, 2, 3, 5, 20, 77, 150, 300):
            chosen = torch.minimum(torch.full_like(self.most, units), self.most).long()
            expected = table[torch.arange(len(self.most)), chosen - 1]
            self.assertTrue(torch.allclose(log_probability(self.curves, self.most, self.needed, chosen).exp(), expected, atol=1e-6))

    def test_draws_follow_the_distribution(self):
        draws = torch.stack([sample(self.curves, self.most, self.needed, torch.Generator().manual_seed(seed)) for seed in range(4000)])
        table = on_grid(self.curves, self.most, self.needed, 300)
        for row in (3, 4):  # 7 and 20 units at the source
            most = int(self.most[row])
            counts = torch.bincount(draws[:, row], minlength=most + 1)[1:].float() / len(draws)
            expected = table[row, :most]
            spread = (expected * (1 - expected) / len(draws)).sqrt()
            self.assertTrue(((counts - expected).abs() <= 5 * spread + 1e-3).all(), row)
        self.assertTrue(((draws >= 1) & (draws <= self.most.long())).all())

    def test_sharpened_draws_follow_the_odds_raised_to_the_power(self):
        count = 4000
        curves, most, needed = self.curves[3:5], self.most[3:5], self.needed[3:5]  # 7 and 20 units
        table = on_grid(curves, most, needed, 20)
        for power in (1.0, 2.0):
            many = (curves.expand(count, *curves.shape), most.expand(count, -1), needed.expand(count, -1))
            draws = sample_sharpened(*many, power, torch.Generator().manual_seed(0))
            self.assertTrue(((draws >= 1) & (draws <= most.long())).all())
            for row in range(2):
                counts = torch.bincount(draws[:, row], minlength=int(most[row]) + 1)[1:].float() / count
                expected = table[row, : int(most[row])] ** power
                expected = expected / expected.sum()
                spread = (expected * (1 - expected) / count).sqrt()
                self.assertTrue(((counts - expected).abs() <= 5 * spread + 1e-3).all(), (power, row))

    def test_a_sharpened_draw_where_nothing_can_be_sent_gives_one(self):
        # Where the choice was to wait, the amount is drawn anyway and then not used.
        drawn = sample_sharpened(self.curves[:2], torch.tensor([0.0, 3.0]), self.needed[:2], 2.0)
        self.assertEqual(drawn[0].item(), 1)

    def test_the_two_anchors_hit_exact_amounts(self):
        most, needed = torch.tensor([40.0]), torch.tensor([13.0])
        cases = (
            ((0.0, 1.0, 0.0), 13),  # just enough
            ((0.0, 1.0, 2.0), 15),  # just enough and two spare
            ((1.0, 0.0, 0.0), 40),  # everything
            ((0.5, 0.0, 0.0), 20),  # half
        )
        for (fraction, share, offset), units in cases:
            chance = log_probability(narrow(fraction, share, offset), most, needed, torch.tensor([units])).exp()
            self.assertGreater(chance.item(), 0.99, (fraction, share, offset))
        # A capture the source cannot afford lands on everything it has.
        short = log_probability(narrow(0.0, 1.0, 0.0), torch.tensor([8.0]), needed, torch.tensor([8])).exp()
        self.assertGreater(short.item(), 0.99)

    def test_curves_far_from_any_amount_keep_values_and_gradients_finite(self):
        curves = torch.tensor([[[0.0, 9.0, -9.0, 50.0, -30.0], [0.0, -9.0, 9.0, -50.0, 30.0], [5.0, 0.0, 0.0, 0.0, -30.0]]])
        curves.requires_grad_(True)
        for units in (1, 2, 17, 39, 40):
            value = log_probability(curves, torch.tensor([40.0]), torch.tensor([20.0]), torch.tensor([units]))
            value.sum().backward()
            self.assertTrue(torch.isfinite(value).all(), units)
            self.assertTrue(torch.isfinite(curves.grad).all(), units)
            curves.grad = None

    def test_entropy_follows_its_definition(self):
        table = on_grid(self.curves, self.most, self.needed, 300)
        direct = -(table * torch.log(table.clamp(min=1e-30))).sum(dim=1)
        self.assertTrue(torch.allclose(entropy(self.curves, self.most, self.needed), direct, atol=1e-4))
        # With one unit there is nothing to decide.
        self.assertEqual(entropy(self.curves[:1], self.most[:1], self.needed[:1]).item(), 0.0)

    def test_entropy_on_grids_by_size_matches_one_wide_grid(self):
        # Routes on both sides of every grid size, so every group is used, each padded.
        most = torch.tensor([float(m) for size in GRID_SIZES for m in (size - 1, size, size + 1)] + [1000.0, 3.0, 3.0])
        needed = (most / 3).floor()
        generator = torch.Generator().manual_seed(3)
        curves = (torch.randn(len(most), 3, PARAMETERS, generator=generator) * 2).requires_grad_(True)
        learned = entropy(curves, most, needed)
        (gradient,) = torch.autograd.grad(learned.sum(), curves)
        table = on_grid(curves, most, needed, GRID_LIMIT)
        direct = -(table * torch.log(table.clamp(min=1e-30))).sum(dim=1)
        (expected,) = torch.autograd.grad(direct.sum(), curves)
        self.assertTrue(torch.allclose(learned, direct, atol=1e-5))
        self.assertTrue(torch.allclose(gradient, expected, atol=1e-5))

    def test_untrained_curves_start_on_just_enough_half_and_everything(self):
        curves = amounts.starting_biases()[None]
        table = on_grid(curves, torch.tensor([40.0]), torch.tensor([12.0]), 40)[0]
        likeliest = set((table.argsort(descending=True)[:3] + 1).tolist())
        self.assertEqual(likeliest, {12, 20, 40})


if __name__ == "__main__":
    unittest.main()
