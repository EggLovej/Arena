"""Seeded symmetric graph families for development and held-out tournaments."""
import random
from game_types import BaseDefinition,EdgeDefinition,MapDefinition,PLAYERS
from loading import validate_map


def generated_map(seed,players):
 rng=random.Random(seed)
 per=rng.randint(4,8)
 names=[[f'node-{rng.getrandbits(64):016x}' for _ in range(per)] for _ in range(players)]
 rates=[rng.uniform(.15,2.4) for _ in range(per)]
 units=[rng.randint(1,25) for _ in range(per)]
 rates[0]=rng.uniform(1,2);units[0]=rng.randint(22,40)
 bases=[]
 for p in range(players):
  for i in range(per):
   bases.append(BaseDefinition(names[p][i],PLAYERS[p] if i==0 else 0,units[i],rates[i],rng.randrange(1000),rng.randrange(800)))
 local=[]
 for i in range(1,per):local.append((i,rng.randrange(i),rng.uniform(.4,6.)))
 existing={frozenset((a,b)) for a,b,t in local}
 for a in range(per):
  for b in range(a+1,per):
   if frozenset((a,b)) not in existing and rng.random()<.2:
    local.append((a,b,rng.uniform(.4,6.)))
 edges=[];seen=set()
 def edge(a,b,t):
  key=frozenset((a,b))
  if key not in seen:edges.append(EdgeDefinition(a,b,t));seen.add(key)
 for p in range(players):
  for a,b,t in local:edge(names[p][a],names[p][b],t)
 for i in rng.sample(range(1,per),rng.randint(1,min(3,per-1))):
  t=rng.uniform(.6,7.)
  for p in range(players):edge(names[p][i],names[(p+1)%players][i],t)
 definition=MapDefinition(f'generated-{seed}-{players}',1000,800,tuple(bases),tuple(edges))
 validate_map(definition)
 return definition


def load_case(name):
 if name.startswith('generated:'):
  _,seed,p=name.split(':');return generated_map(int(seed),int(p))
 from loading import load_map
 return load_map(name)
