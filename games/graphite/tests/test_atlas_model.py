"""Mirror's network forecasts of its opponents."""

import unittest

from game import Game
from game_types import RED, BLUE, GREEN
from loading import load_map
from ais import _atlas_model_planner as planner


class LearnedPlanningTests(unittest.TestCase):
    def test_model_loading_and_sampling_preserve_global_rng(self):
        import torch
        planner._MODEL=None  # loaded again here, even if an earlier test loaded it
        before=torch.random.get_rng_state().clone()
        planner.model()
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        views=[Game(load_map('triangle')).view_for(p) for p in (RED,BLUE,GREEN)]
        first=planner.predict(views,torch.Generator().manual_seed(137))
        second=planner.predict(views,torch.Generator().manual_seed(137))
        self.assertEqual(first,second)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))

    def test_sampled_actions_remain_legal_in_the_real_engine(self):
        import torch
        game=Game(load_map('triangle'))
        generator=torch.Generator().manual_seed(843)
        for _ in range(35):
            views=[game.view_for(p) for p in game.players]
            snapshots=tuple(views)
            actions=planner.predict(views,generator)
            self.assertEqual(tuple(game.view_for(p) for p in game.players),snapshots)
            self.assertFalse(game.step(dict(zip(game.players,actions))))


if __name__=='__main__':unittest.main()
