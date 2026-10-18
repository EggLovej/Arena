import unittest
from unittest.mock import patch

from game import Game
from game_types import RED, BLUE, GREEN, BaseDefinition, EdgeDefinition, MapDefinition, SendAction
from loading import load_map
from ais import aggressive_ai
from ais.atlas_ai import policy
from ais import _atlas_model_planner as planner


class LearnedPlanningTests(unittest.TestCase):
    def test_model_loading_and_sampling_preserve_global_rng(self):
        import torch
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

    def test_impossible_encoded_amount_rules_out_learned_hypothesis(self):
        from ml.encoding.action_encoder import legal_actions
        state=Game(load_map('triangle')).view_for(RED)
        legal=legal_actions(state)
        for i,source in enumerate(state.bases):
            if source.owner!=RED:continue
            for j,target in enumerate(state.bases):
                if not legal.allowed[i,j].any():continue
                allowed=set(int(x) for x in legal.units[i,j] if x>0)
                missing=next((x for x in range(1,int(source.units)+1) if x not in allowed),None)
                if missing is not None:
                    action=SendAction(source.id,target.id,missing)
                    self.assertIsNone(planner.evidence(state,action,[None]*13))
                    return
        self.fail('Fixture needs a legal army amount outside the neural vocabulary')

    def test_mixed_rollouts_sample_only_the_learned_opponent(self):
        state=Game(load_map('triangle')).view_for(RED)
        seen=[]
        original=planner.predict
        def record(views,*args,**kwargs):
            seen.extend(v.player_id for v in views)
            return original(views,*args,**kwargs)
        with patch.object(planner,'HORIZON',.2), patch.object(planner,'SAMPLES',1), patch.object(planner,'CANDIDATES',[planner.CANDIDATES[0]]), patch.object(planner,'predict',side_effect=record):
            selected=planner.select(state,{BLUE:aggressive_ai.choose_action,GREEN:None})
        self.assertEqual(selected,planner.CANDIDATES[0])
        self.assertTrue(seen)
        self.assertEqual(set(seen),{GREEN})

    def test_campaign_funding_enables_the_configured_attack(self):
        definition=MapDefinition('funding',300,100,(
            BaseDefinition('rear',RED,20,0,0,0),
            BaseDefinition('front',RED,15,0,100,0),
            BaseDefinition('enemy',BLUE,10,0,200,0),
        ),(EdgeDefinition('rear','front',1.),EdgeDefinition('front','enemy',1.)))
        game=Game(definition)
        config=dict(overwhelm=2.,reserve=0.,buffer=0.,holding=0.,saving=False)
        self.assertIsNone(policy(game.view_for(RED),**config))
        action=policy(game.view_for(RED),campaign_funding=True,**config)
        self.assertEqual(action,SendAction('rear','front',10))
        self.assertFalse(game.step({RED:action,BLUE:None}))
        for _ in range(9):game.step({RED:None,BLUE:None})
        attack=policy(game.view_for(RED),campaign_funding=True,**config)
        self.assertEqual((attack.source_id,attack.target_id),('front','enemy'))
        self.assertGreaterEqual(attack.units,22)
        game.step({RED:attack,BLUE:None})
        for _ in range(9):game.step({RED:None,BLUE:None})
        self.assertEqual(game.winner(),RED)


if __name__=='__main__':unittest.main()
