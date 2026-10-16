"""Differential tests against the real engine, not an alternate mock of its rules."""
import unittest
import random
from dataclasses import replace
from game import Game
from game_types import *
from loading import load_map
from ais._atlas_policy import Position,arrival_ticks
from ais._atlas_sim import Simulation

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

 def test_batched_simulator_matches_every_tick(self):
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

 def test_capture_requirement_is_minimum_with_three_way_arrival(self):
  definition=MapDefinition('event',100,100,(
   BaseDefinition('a',RED,100,0,0,0),BaseDefinition('b',BLUE,10,0,30,0),
   BaseDefinition('c',GREEN,100,0,60,0)),(
   EdgeDefinition('a','b',.3),EdgeDefinition('c','b',.3)))
  from game import Fleet
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
  from game import Fleet
  definition=MapDefinition('reserve',100,100,(
   BaseDefinition('a',RED,25,1,0,0),BaseDefinition('b',BLUE,50,0,30,0)),(EdgeDefinition('a','b',5),))
  g=Game(definition);g.fleets=[Fleet(BLUE,12,'b','a',2,5),Fleet(BLUE,15,'b','a',4,5)]
  p=Position(g.view_for(RED));available=p.available(0)
  self.assertEqual(available,1)
  g.step({RED:SendAction('a','b',available)})
  for _ in range(40):
   self.assertEqual(g.bases['a'].owner,RED);g.step({})


class ControllerTests(unittest.TestCase):
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

 def test_model_fits_visible_departures(self):
  from ais.atlas_ai import choose_action,_CONTEXTS
  from ais.careful_ai import choose_action as opponent
  g=Game(load_map('classic'))
  for tick in range(25):
   action=choose_action(g.view_for(RED))
   self.assertFalse(g.step({RED:action,BLUE:opponent(g.view_for(BLUE))}))
  ctx=next(iter(_CONTEXTS.values()))
  errors=ctx['errors'][BLUE]
  self.assertEqual(errors[1],0.)
  self.assertGreater(errors[0],0.)
  self.assertGreater(errors[2],0.)

 def test_arbitrary_ids_and_coordinates(self):
  from ais.atlas_ai import choose_action
  g=Game(load_map('crossroads'));state=g.view_for(RED)
  action=choose_action(state)
  ids={b.id:f'unseen-{137*i+19}' for i,b in enumerate(state.bases)}
  transformed=replace(state,
   bases=tuple(replace(b,id=ids[b.id],x=3,y=9) for b in state.bases),
   edges=tuple(replace(e,a=ids[e.a],b=ids[e.b]) for e in state.edges))
  other=choose_action(transformed)
  self.assertEqual(other,SendAction(ids[action.source_id],ids[action.target_id],action.units))

 def test_nonstandard_tick_durations_and_same_module_players(self):
  from ais.atlas_ai import choose_action
  for dt in (.13,.7,3.):
   g=Game(load_map('triangle'),dt)
   for tick in range(3):
    actions={p:choose_action(g.view_for(p)) for p in g.players}
    self.assertFalse(g.step(actions))

class ContinuationTests(unittest.TestCase):
 def test_opponent_trajectories_match_without_float_tolerance(self):
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

 def test_large_armies_are_not_terminal_states(self):
  g=Game(load_map('classic'))
  g.bases['west_home'].units=10**8
  sim=Simulation(g.view_for(RED))
  self.assertFalse(g.is_finished())
  self.assertLess(abs(sim.value(RED)),100000)

 def test_mismatched_observation_invalidates_continuation(self):
  from ais.atlas_ai import choose_action,_CONTEXTS
  from ais.careful_ai import choose_action as opponent
  _CONTEXTS.clear()
  g=Game(load_map('classic'))
  action=choose_action(g.view_for(RED))
  ctx=next(iter(_CONTEXTS.values()))
  ctx['certified']={BLUE:lambda view:None}
  ctx['certify_after']=10**9
  enemy_action=opponent(g.view_for(BLUE))
  self.assertIsNotNone(enemy_action)
  self.assertFalse(g.step({RED:action,BLUE:enemy_action}))
  choose_action(g.view_for(RED))
  self.assertIsNone(ctx['certified'])

 def test_predicted_winning_continuation_finishes_in_real_engine(self):
  from ais.atlas_ai import choose_action,_CONTEXTS
  from ais.strategic_ai import choose_action as opponent
  _CONTEXTS.clear()
  g=Game(load_map('classic'));checked=False
  for _ in range(1800):
   action=choose_action(g.view_for(RED))
   checked |= bool(next(iter(_CONTEXTS.values()))['certified'])
   self.assertFalse(g.step({RED:action,BLUE:opponent(g.view_for(BLUE))}))
   if g.is_finished():break
  self.assertTrue(checked)
  self.assertEqual(g.winner(),RED)

 def test_eliminated_candidate_is_a_loss_before_overall_match_ends(self):
  from benchmark_ai import outcome_for
  g=Game(load_map('triangle'))
  self.assertEqual(outcome_for(g,RED),'timeout')
  for base in g.bases.values():
   if base.owner==RED:base.owner=BLUE
  self.assertFalse(g.is_finished())
  self.assertEqual(outcome_for(g,RED),'loss')


class EvaluatorTests(unittest.TestCase):
 def test_existing_evaluator_counts_eliminated_seat_as_loss(self):
  import io
  from contextlib import redirect_stdout
  from unittest.mock import patch
  import evaluate
  definition=MapDefinition('evaluation',100,100,(
   BaseDefinition('red',RED,0,0,0,0),
   BaseDefinition('blue',BLUE,10,0,50,0),
   BaseDefinition('green',GREEN,10,0,100,0)),(
   EdgeDefinition('red','blue',.1),EdgeDefinition('blue','green',1)))
  def enemy(state):
   if state.player_id==BLUE and state.get_base('red').owner==RED:
    return SendAction('blue','red',1)
   return None
  def load(name):return (lambda state:None) if name=='candidate' else enemy
  with patch.object(evaluate,'available_maps',return_value=('scenario',)),patch.object(evaluate,'load_map',return_value=definition),patch.object(evaluate,'load_ai',side_effect=load),redirect_stdout(io.StringIO()):
   self.assertEqual(evaluate.evaluate('candidate','opponent',.1,.1),(1.,3))

if __name__=='__main__':unittest.main()
