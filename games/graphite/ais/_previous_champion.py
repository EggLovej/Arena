"""Experimental opponent-adaptive policy portfolio."""
from dataclasses import replace
from functools import partial
from collections import OrderedDict
from ais._atlas_policy import policy
from ais._atlas_reference_sim import Simulation
from ais import aggressive_ai,careful_ai,strategic_ai

CONFIGS = [
 dict(reserve=0.,buffer=0.,saving=False,holding=0.,costscale=20.,surge=1.,enemy=3.),
 dict(reserve=.5,buffer=0.,saving=False,holding=0.,costscale=10.,surge=.9),
 dict(reserve=1.,buffer=0.,saving=False,holding=0.,costscale=10.,surge=.6),
 dict(reserve=1.,buffer=0.,saving=False,holding=0.,costscale=5.),
 dict(reserve=1.,buffer=0.,saving=False,holding=.65,costscale=5.),
 dict(reserve=.5,buffer=0.,saving=False,holding=.25,costscale=20.),
 dict(reserve=1.,buffer=0.,saving=True,holding=.25,costscale=10.,horizon=40.),
 dict(reserve=2.,buffer=2.,saving=True,holding=.5,costscale=20.),
]
MODELS=[strategic_ai.choose_action,careful_ai.choose_action,aggressive_ai.choose_action,
        partial(policy,**CONFIGS[0]),partial(policy,**CONFIGS[1])]
_CONTEXTS=OrderedDict()

def action_distance(a,b):
 if a==b:return 0.
 if a is None or b is None:return 1.
 if (a.source_id,a.target_id)!=(b.source_id,b.target_id):return 1.
 return min(1.,abs(a.units-b.units)/max(1,a.units,b.units))


def choose_action(state):
 from game_types import SendAction
 key=(state.player_id,state.seconds_per_tick,tuple((b.id,b.production_rate) for b in state.bases),state.edges)
 ctx=_CONTEXTS.get(key)
 if ctx and ctx['previous']==state:return ctx['action']
 if ctx is None or state.tick!=ctx['previous'].tick+1:
  ctx=dict(previous=state,errors={},config=CONFIGS[0],next_plan=state.tick,action=None)
  _CONTEXTS[key]=ctx
  while len(_CONTEXTS)>12:_CONTEXTS.popitem(last=False)
 else:
  prev=ctx['previous']
  for enemy in set(b.owner for b in prev.bases)-{0,state.player_id}:
   # Fleets that departed in the previous tick are publicly observable.
   if min((e.travel_time for e in state.edges),default=0)<=state.seconds_per_tick:continue
   recent=[f for f in state.fleets if f.owner==enemy and abs(f.total_time-f.remaining_time-state.seconds_per_tick)<1e-7]
   actual=SendAction(recent[0].source_id,recent[0].target_id,recent[0].units) if recent else None
   view=replace(prev,player_id=enemy)
   estimates=[model(view) for model in MODELS]
   if actual is not None or any(a is not None for a in estimates):
    errors=ctx['errors'].setdefault(enemy,[0.]*len(MODELS))
    for i,a in enumerate(estimates):errors[i]=errors[i]*.97+action_distance(a,actual)
 if state.tick>=ctx['next_plan']:
  players=sorted(set(b.owner for b in state.bases)-{0} | {f.owner for f in state.fleets})
  models={p:MODELS[min(range(len(MODELS)),key=lambda i:ctx['errors'].get(p,[0.]*len(MODELS))[i])] for p in players if p!=state.player_id}
  step=max(1,round(.5/state.seconds_per_tick))
  durations=[1]*max(1,round(8/state.seconds_per_tick))+[step]*max(1,round(22/(step*state.seconds_per_tick)))
  nsteps=len(durations)
  best=(-float('inf'),ctx['config'])
  for cfg in CONFIGS:
   sim=Simulation(state);integral=0
   for k,duration in enumerate(durations):
    actions={p:policy(sim.view(p),**cfg) if p==state.player_id else models[p](sim.view(p)) for p in players}
    sim.step(actions,duration)
    value=sim.value(state.player_id)
    if abs(value)>1e5:
     value += (1 if value>0 else -1)*(30-sim.tick*state.seconds_per_tick)*100
     break
    integral+=value/nsteps
   score=value+.1*integral
   if score>best[0]:best=(score,cfg)
  ctx['config']=best[1]
  ctx['next_plan']=state.tick+max(1,round(1./state.seconds_per_tick))
 action=policy(state,**ctx['config'])
 ctx['previous']=state;ctx['action']=action
 return action
