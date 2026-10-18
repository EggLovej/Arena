import unittest
from dataclasses import replace

import numpy as np
import torch

from ml.map_generator import generate_map
from ml.play.rollout import Matchup, Seat, play_games
from ml.tests.support import trajectory_choices, trajectory_input
from ml.training.dataset import DecisionSet


def teacher_trajectories(seeds=range(5)):
    matchups = [
        Matchup(generate_map(seed).definition, Seat("strategic_ai", record=True), Seat("aggressive_ai"))
        for seed in seeds
    ]
    return [record.trajectories[0] for record in play_games(matchups, max_seconds=40)]


class DecisionSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajectories = teacher_trajectories()

    def full_set(self):
        decisions = DecisionSet()
        for number, trajectory in enumerate(self.trajectories):
            decisions.add(trajectory, {"game": np.full(len(trajectory), number), "tick": trajectory.tick})
        return decisions

    def test_one_pass_visits_every_decision_exactly_once_shuffled_or_not(self):
        decisions = self.full_set()
        self.assertEqual(len(decisions), sum(len(trajectory) for trajectory in self.trajectories))
        self.assertEqual(len(DecisionSet()), 0)

        def visited(rng):
            seen = []
            for batch in decisions.batches(64, rng):
                self.assertLessEqual(len(batch), 64)
                seen += list(zip(batch.targets["game"].tolist(), batch.targets["tick"].tolist()))
            return seen

        everything = [
            (float(number), float(tick))
            for number, trajectory in enumerate(self.trajectories)
            for tick in trajectory.tick
        ]
        in_order, shuffled = visited(None), visited(np.random.default_rng(0))
        self.assertEqual(sorted(in_order), sorted(everything))
        self.assertEqual(sorted(shuffled), sorted(everything))
        self.assertNotEqual(in_order, shuffled)

    def test_route_features_survive_maps_with_different_route_counts(self):
        # Maps with the same number of bases share a group even when their
        # route counts differ, so the stored route lists are padded.
        # A copy of one game that lost its last two routes stands in for such a map.
        first = self.trajectories[0]
        fewer = replace(first, routes=first.routes[:-2], route_features=first.route_features[:, :-2])
        trajectories = [first, fewer]
        decisions = DecisionSet()
        for number, trajectory in enumerate(trajectories):
            decisions.add(trajectory, {"game": np.full(len(trajectory), number), "decision": np.arange(len(trajectory))})
        for batch in decisions.batches(64, np.random.default_rng(0)):
            for row, (game, decision) in enumerate(zip(batch.targets["game"].int().tolist(), batch.targets["decision"].int().tolist())):
                expected = trajectories[game].route_table([decision])[0]
                np.testing.assert_array_equal(batch.inputs.route_features[row].numpy(), expected)

    def test_each_batch_holds_one_map_size_and_no_padding(self):
        sizes = {len(trajectory.base_ids) for trajectory in self.trajectories}
        self.assertGreater(len(sizes), 1)
        for batch in self.full_set().batches(64, np.random.default_rng(0)):
            self.assertIn(batch.inputs.base_mask.shape[1], sizes)
            self.assertTrue(batch.inputs.base_mask.all())

    def test_batches_match_the_trajectory_they_came_from(self):
        trajectory = self.trajectories[0]
        decisions = DecisionSet()
        decisions.add(trajectory, {"tick": trajectory.tick})
        batch = next(decisions.batches(10_000))

        expected = trajectory_input(trajectory)
        route, amount = trajectory_choices(trajectory)
        self.assertTrue(torch.equal(batch.inputs.base_features, expected.base_features))
        self.assertTrue(torch.equal(batch.inputs.pair_features, expected.pair_features))
        self.assertTrue(torch.equal(batch.inputs.route_features, expected.route_features))
        self.assertTrue(torch.equal(batch.inputs.global_features, expected.global_features))
        self.assertTrue(torch.equal(batch.inputs.allowed, expected.allowed))
        self.assertTrue(torch.equal(batch.route, route))
        self.assertTrue(torch.equal(batch.amount, amount))
        self.assertEqual(batch.targets["tick"].tolist(), trajectory.tick.tolist())

    def test_every_stored_choice_is_legal(self):
        for batch in self.full_set().batches(128):
            sends = batch.route != 0
            bases = batch.inputs.base_mask.shape[1]
            source, target = torch.div(batch.route - 1, bases, rounding_mode="floor"), (batch.route - 1) % bases
            rows = torch.arange(len(batch))[sends]
            self.assertTrue(batch.inputs.allowed[rows, source[sends], target[sends], batch.amount[sends]].all())

    def test_long_trajectories_can_be_sampled_down(self):
        trajectory = max(self.trajectories, key=len)
        self.assertGreater(len(trajectory), 50)
        decisions = DecisionSet()
        added = decisions.add(trajectory, {"tick": trajectory.tick}, max_decisions=50, rng=np.random.default_rng(1))
        self.assertEqual((added, len(decisions)), (50, 50))

        ticks = next(decisions.batches(100)).targets["tick"].tolist()
        self.assertEqual(ticks, sorted(set(ticks)))
        self.assertTrue(set(ticks) <= set(trajectory.tick.tolist()))

        short = DecisionSet()
        self.assertEqual(short.add(trajectory, max_decisions=10_000), len(trajectory))

    def test_equal_size_batches_all_have_the_full_size_and_miss_nothing(self):
        decisions = self.full_set()
        seen = set()
        for batch in decisions.batches(64, np.random.default_rng(0), equal_size=True):
            self.assertEqual(len(batch), 64)
            seen |= set(zip(batch.targets["game"].tolist(), batch.targets["tick"].tolist()))
        everything = {
            (float(number), float(tick))
            for number, trajectory in enumerate(self.trajectories)
            for tick in trajectory.tick
        }
        self.assertEqual(seen, everything)

        # A group smaller than one batch still fills one.
        small = DecisionSet()
        trajectory = self.trajectories[0]
        small.add(trajectory, {"tick": trajectory.tick}, max_decisions=5, rng=np.random.default_rng(0))
        self.assertEqual([len(batch) for batch in small.batches(32, np.random.default_rng(0), equal_size=True)], [32])

    def test_targets_must_be_the_same_for_every_trajectory(self):
        decisions = DecisionSet()
        first, second = self.trajectories[:2]
        decisions.add(first, {"outcome": np.zeros(len(first))})
        with self.assertRaises(ValueError):
            # Same map size as an earlier trajectory but different targets.
            decisions.add(first, {"advantage": np.zeros(len(first))})


class FreeAmountSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajectory = teacher_trajectories(seeds=[2])[0]

    def test_a_free_set_learns_the_units_and_carries_every_routes_limits(self):
        trajectory = self.trajectory
        decisions = DecisionSet(free_amounts=True, option_units=True)
        decisions.add(trajectory)
        # Without a random generator the stored order is kept, so one batch is the whole trajectory.
        (batch,) = list(decisions.batches(len(trajectory)))
        sent = batch.route > 0
        np.testing.assert_array_equal(batch.amount[sent].numpy(), trajectory.units[trajectory.source >= 0])
        limits = trajectory.limit_table()
        np.testing.assert_array_equal(batch.inputs.most.numpy(), limits[..., 0])
        np.testing.assert_array_equal(batch.inputs.needed.numpy(), limits[..., 1])
        # What every option sends, rebuilt from the limits, is on offer exactly where it was.
        self.assertTrue(torch.equal(batch.option_units > 0, batch.inputs.allowed))
        self.assertTrue((batch.inputs.most[batch.inputs.allowed.any(dim=-1)] >= 1).all())

    def test_an_options_set_leaves_the_limits_out(self):
        decisions = DecisionSet()
        decisions.add(self.trajectory)
        batch = next(decisions.batches(64))
        self.assertIsNone(batch.inputs.most)
        self.assertIsNone(batch.option_units)

    def test_an_options_set_refuses_choices_made_with_free_amounts(self):
        with self.assertRaises(ValueError):
            DecisionSet().add(replace(self.trajectory, free_amounts=True))
        # And a free set needs the limits, which older trajectories lack.
        with self.assertRaises(ValueError):
            DecisionSet(free_amounts=True).add(replace(self.trajectory, route_limits=None))


if __name__ == "__main__":
    unittest.main()
