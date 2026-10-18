"""Atlas's forecasts and policy, and Mirror's play, tested against the real engine, not a mock of its rules."""
import unittest
import random
from dataclasses import replace
from game import Fleet, Game
from game_types import BLUE, GREEN, RED, BaseDefinition, EdgeDefinition, MapDefinition, SendAction
from loading import load_map
from ais._atlas_policy import Position
from ais._atlas_sim import Simulation
from ais.atlas_clean_ai import policy

class ForecastTests(unittest.TestCase):
 def random_game(self,seed):
  rng=random.Random(seed)
  g=Game(load_map('triangle'),seconds_per_tick=rng.choice([.1,.13,.25,.7]))
  for _ in range(30):
   acts={}
   for p in g.players:
    mine=[b for b in g.bases.values() if b.owner==p and int(b.units)>0]
    if mine and rng.random()<.7:
     b=rng.choice(mine);neighbors=g.view_for(p).neighbors(b.id)
     acts[p]=SendAction(b.id,rng.choice(neighbors).id,rng.randint(1,int(b.units)))
   g.step(acts)
  return g

 def test_event_projection_matches_engine(self):
  for seed in range(100):
   g=self.random_game(seed);p=Position(g.view_for(RED))
   for t in range(1,30):
    g.step({})
    for i,b in enumerate(p.bases):
     owner,units=p.project(i,t)
     self.assertEqual(owner,g.bases[b.id].owner,(seed,t,b.id))
     self.assertAlmostEqual(units,g.bases[b.id].units,places=7)

 def test_simulator_matches_the_engine(self):
  for seed in range(60):
   g=self.random_game(seed);sim=Simulation(g.view_for(RED));rng=random.Random(seed)
   for cycle in range(12):
    actions={}
    for player in g.players:
     mine=[b for b in g.bases.values() if b.owner==player and int(b.units)]
     if mine:
      b=rng.choice(mine);target=rng.choice(g.view_for(player).neighbors(b.id))
      actions[player]=SendAction(b.id,target.id,rng.randint(1,int(b.units)))
    n=rng.randint(1,9);sim.step(actions,n);g.step(actions)
    for _ in range(n-1):g.step({})
    view=sim.view(RED)
    for a,b in zip(view.bases,g.view_for(RED).bases):
     self.assertEqual(a.owner,b.owner)
     self.assertAlmostEqual(a.units,b.units,places=7)
    key=lambda f:(f.owner,f.source_id,f.target_id,f.units,f.total_time)
    a=sorted(view.fleets,key=key);b=sorted(g.view_for(RED).fleets,key=key)
    self.assertEqual([key(f) for f in a],[key(f) for f in b])
    for x,y in zip(a,b):self.assertAlmostEqual(x.remaining_time,y.remaining_time,places=7)
  # Tick by tick it matches exactly, so forecast opponents play exactly as the real ones do.
  from ais.strategic_ai import choose_action
  for name in ('classic','ring','triangle','three_fronts'):
   g=Game(load_map(name))
   for _ in range(31):
    g.step({p:choose_action(g.view_for(p)) for p in g.players})
   sim=Simulation(g.view_for(RED))
   for _ in range(300):
    actual={p:choose_action(g.view_for(p)) for p in g.players}
    predicted={p:choose_action(sim.view(p)) for p in g.players}
    self.assertEqual(actual,predicted)
    g.step(actual);sim.step(predicted,1)
    self.assertEqual(g.view_for(RED),sim.view(RED))

 def test_capture_requirement_is_minimum_with_three_way_arrival(self):
  definition=MapDefinition('event',100,100,(
   BaseDefinition('a',RED,100,0,0,0),BaseDefinition('b',BLUE,10,0,30,0),
   BaseDefinition('c',GREEN,100,0,60,0)),(
   EdgeDefinition('a','b',.3),EdgeDefinition('c','b',.3)))
  g=Game(definition)
  g.fleets=[Fleet(GREEN,25,'c','b',.3,.3),Fleet(RED,4,'a','b',.3,.3)]
  p=Position(g.view_for(RED));need=p.required(1,3)
  self.assertEqual(need,22)  # max(10,25) - 4 + 1, NOT 10 + 25 - 4 + 1.
  for amount in [need-1,need]:
   h=Game(definition);h.fleets=list(g.fleets)
   # Fresh fleet instances because the real engine mutates them.
   h.fleets=[replace(f) for f in g.fleets]
   h.step({RED:SendAction('a','b',amount)})
   h.step({});h.step({})
   self.assertEqual(h.bases['b'].owner==RED,amount==need)

 def test_reserve_protects_all_committed_arrivals(self):
  definition=MapDefinition('reserve',100,100,(
   BaseDefinition('a',RED,25,1,0,0),BaseDefinition('b',BLUE,50,0,30,0)),(EdgeDefinition('a','b',5),))
  g=Game(definition);g.fleets=[Fleet(BLUE,12,'b','a',2,5),Fleet(BLUE,15,'b','a',4,5)]
  p=Position(g.view_for(RED));available=p.available(0)
  self.assertEqual(available,1)
  g.step({RED:SendAction('a','b',available)})
  for _ in range(40):
   self.assertEqual(g.bases['a'].owner,RED);g.step({})

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



class MirrorTests(unittest.TestCase):
 def setUp(self):
  from ais.atlas_ai import _CONTEXTS
  _CONTEXTS.clear()

 def test_replay_duplicate_observations_and_legality(self):
  from ais.atlas_ai import choose_action
  from ais.strategic_ai import choose_action as opponent
  final=[]
  for replay in range(2):
   g=Game(load_map('ring'))
   for tick in range(35):
    before=g.view_for(RED)
    action=choose_action(before)
    self.assertEqual(action,choose_action(before))
    self.assertEqual(before,g.view_for(RED))
    errors=g.step({RED:action,BLUE:opponent(g.view_for(BLUE))})
    self.assertFalse(errors)
   final.append(g.view_for(RED))
  self.assertEqual(*final)

 def test_predicted_winning_continuation_finishes_in_real_engine(self):
  from ais.atlas_ai import choose_action,_CONTEXTS
  from ais.strategic_ai import choose_action as opponent
  g=Game(load_map('classic'));checked=False
  for _ in range(1800):
   action=choose_action(g.view_for(RED))
   checked |= bool(next(iter(_CONTEXTS.values()))['certified'])
   self.assertFalse(g.step({RED:action,BLUE:opponent(g.view_for(BLUE))}))
   if g.is_finished():break
  self.assertTrue(checked)
  self.assertEqual(g.winner(),RED)

if __name__=='__main__':unittest.main()
