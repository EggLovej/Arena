"""Atlas's batched stochastic planner and learned-opponent likelihood model.

Loads an independent public checkpoint; never reads an opponent instance or RNG.
The learned network predicts enemy actions. Our continuation policies remain
explicit tactical policies; leaf evaluation uses game strength, not its value
head. Shared random draws help compare policies. No global optimality is claimed.
"""
from dataclasses import replace
from pathlib import Path
from math import sqrt
from ais._atlas_sim import Simulation
from ais.atlas_clean_ai import STYLES, policy as ordinary_policy

CANDIDATES = [
    dict(STYLES[0]), dict(STYLES[3]), dict(STYLES[5]),
    dict(STYLES[6], campaign=True), dict(STYLES[8], campaign=True),
    dict(horizon=50., reserve=2., buffer=1., holding=.15, costscale=3.,
         enemy=8., expansion=4., saving=True, ffa=True, surge=.3, rally=2., overwhelm=1., campaign=True),
]
CANDIDATES.extend([{'horizon': 50.0,
  'reserve': 4.0,
  'buffer': 8.0,
  'holding': 0.8,
  'costscale': 10.0,
  'enemy': 8.0,
  'expansion': 2.0,
  'saving': True,
  'ffa': False,
  'surge': 0.0,
  'rally': 0.0,
  'overwhelm': 1.0},
 {'horizon': 35.0,
  'reserve': 8.0,
  'buffer': 4.0,
  'holding': 1.5,
  'costscale': 1.0,
  'enemy': 5.0,
  'expansion': 1.0,
  'saving': False,
  'ffa': True,
  'surge': 0.3,
  'rally': 2.0,
  'overwhelm': 2.0},
 {'horizon': 18.0,
  'reserve': 0.7,
  'buffer': 2.0,
  'holding': 0.8,
  'costscale': 80.0,
  'enemy': 2.0,
  'expansion': 0.5,
  'saving': True,
  'ffa': True,
  'surge': 1.0,
  'rally': 1.0,
  'overwhelm': 1.5}])
_MODEL = None
_CHECKPOINT_BYTES = None
CHECKPOINT_INFO = None

def configure_checkpoint(path=None):
    """Pin the public checkpoint for the new match; the network is loaded again when the file changed."""
    global _MODEL, _CHECKPOINT_BYTES, CHECKPOINT_INFO
    import os, hashlib
    root=Path(__file__).resolve().parent.parent
    default=root/'ml/runs/latest.pt'
    if not default.exists():default=root/'checkpoints/neural_ai.pt'
    selected=Path(path or os.environ.get('ATLAS_OPPONENT_CHECKPOINT',os.environ.get('NEURAL_AI_CHECKPOINT',str(default))))
    data=selected.read_bytes()
    digest=hashlib.sha256(data).hexdigest()
    if CHECKPOINT_INFO is None or digest!=CHECKPOINT_INFO['sha256']:
        _MODEL=None
    _CHECKPOINT_BYTES=data
    CHECKPOINT_INFO=dict(path=str(selected.resolve()),sha256=digest)
    return dict(CHECKPOINT_INFO)

HORIZON = 60.
SAMPLES = 3

def tactical(state, config):
    cfg=dict(config)
    cfg['campaign_funding']=cfg.pop('campaign', False)
    return ordinary_policy(state, **cfg)

def model():
    global _MODEL
    if _MODEL is None:
        import torch
        from ml.models.checkpoint import load_checkpoint
        torch.set_num_threads(1)
        if _CHECKPOINT_BYTES is None:configure_checkpoint()
        with torch.random.fork_rng(devices=[]):
            _MODEL=load_checkpoint(CHECKPOINT_INFO['path'],data=_CHECKPOINT_BYTES).model.eval()
    return _MODEL

def known(amounts):
    """Only the amount options the network was trained with. ml lists more, which it never picks."""
    return amounts[...,:model().config.amount_options]

def predict(views, generator=None, uniforms=None):
    import torch
    from ml.encoding.action_encoder import legal_actions, decode_action
    from ml.encoding.state_encoder import encode_state, layout_for
    from ml.models import make_input
    from ml.models.policy import choose, to_choices
    layouts=[layout_for(v) for v in views]
    from ml.encoding.forecast import build_timelines
    worlds={}; timelines=[]
    for v in views:
        key=(id(v.bases),id(v.fleets),v.seconds_per_tick)
        if key not in worlds:worlds[key]=build_timelines(v)
        timelines.append(worlds[key])
    legal=[legal_actions(v,l,t) for v,l,t in zip(views,layouts,timelines)]
    inputs=make_input([encode_state(v,l,t) for v,l,t in zip(views,layouts,timelines)],legal)
    with torch.inference_mode():
        output=model()(inputs)
        output=replace(output,amount_logits=known(output.amount_logits))
        if uniforms is None:
            route,amount=choose(output,generator=generator)
        else:
            def pick(logits, u):
                cdf=torch.softmax(logits,dim=1).cumsum(dim=1)
                cdf=cdf/cdf[:,-1:]
                return torch.searchsorted(cdf.contiguous(),u[:,None].contiguous(),right=True)[:,0]
            route=pick(output.route_logits,uniforms[:,0])
            pair=(route-1).clamp(min=0)
            logits=output.amount_logits.flatten(start_dim=1,end_dim=2)[torch.arange(len(route)),pair]
            amount=pick(logits,uniforms[:,1])
        choices=to_choices(route,amount,bases=inputs.base_mask.shape[1])
    return [decode_action(c,l) for c,l in zip(choices,legal)]

def select(state, opponents=None):
    import torch
    generator=torch.Generator().manual_seed(170003 + state.tick*131 + state.player_id*17)
    simulations=[Simulation(state) for _ in range(len(CANDIDATES)*SAMPLES)]
    terminal={}
    players=sorted(set(b.owner for b in state.bases)-{0} | {f.owner for f in state.fleets})
    for step in range(max(1,round(HORIZON/state.seconds_per_tick))):
        views=[];keys=[];orders={}
        for i,sim in enumerate(simulations):
            if i in terminal:continue
            own=sim.view(state.player_id)
            orders[i]={state.player_id:tactical(own,CANDIDATES[i//SAMPLES])}
            for p in players:
                if p==state.player_id:continue
                v=sim.view(p)
                if opponents is not None and opponents.get(p) is not None:
                    orders[i][p]=opponents[p](v)
                    continue
                if any(b.owner==p and b.units>=1 for b in v.bases):
                    views.append(v);keys.append((i,p))
                else:orders[i][p]=None
        if not orders:break
        if views:
            # Common random numbers reduce comparison noise between policies.
            draws=torch.rand(SAMPLES,len(players),2,generator=generator)
            uniforms=torch.stack([draws[i%SAMPLES,players.index(p)] for i,p in keys])
            for (i,p),action in zip(keys,predict(views,generator,uniforms=uniforms)):
                orders[i][p]=action
        for i,actions in orders.items():
            sim=simulations[i];sim.step(actions,1)
            alive=set(sim.owners)-{0} | {f.owner for _,f in sim.fleets}
            if state.player_id not in alive:terminal[i]=-1.
            elif alive=={state.player_id}:terminal[i]=1.
    nonterminal=[i for i in range(len(simulations)) if i not in terminal]
    values=dict(terminal)
    for i in nonterminal:
        # Material and production, squashed below the terminal values.
        heuristic=simulations[i].value(state.player_id)
        values[i]=.99*heuristic/(100+abs(heuristic))
    scores=[]
    for c in range(len(CANDIDATES)):
        vals=[values[c*SAMPLES+j] for j in range(SAMPLES)]
        mean=sum(vals)/len(vals)
        std=sqrt(sum((x-mean)**2 for x in vals)/len(vals))
        scores.append(mean-.15*std)
    return dict(CANDIDATES[max(range(len(scores)),key=scores.__getitem__)])


def evidence(state, action, predictions):
    """Likelihood relative to a smoothed mixture of generic public policies.

    Returns None for an action outside the learned model's action vocabulary.
    No actual opponent object, chosen controller name, or random state is used.
    """
    import torch
    from math import log
    from ml.encoding.action_encoder import legal_actions
    from ml.encoding.state_encoder import encode_state, layout_for
    from ml.models import make_input
    layout=layout_for(state)
    from ml.encoding.forecast import build_timelines
    timelines=build_timelines(state)
    legal=legal_actions(state,layout,timelines)
    units=known(legal.units)
    if action is not None:
        i,j=layout.index[action.source_id],layout.index[action.target_id]
        matching=(units[i,j]==action.units) & known(legal.allowed)[i,j]
        if not matching.any():return None
    with torch.inference_mode():
        out=model()(make_input([encode_state(state,layout,timelines)],[legal]))
        routes=torch.softmax(out.route_logits[0],dim=0)
        if action is None:
            probability=float(routes[0])
        else:
            amounts=torch.softmax(known(out.amount_logits[0,i,j]),dim=0)
            probability=float(routes[1+i*len(layout.base_ids)+j]*amounts[torch.from_numpy(matching)].sum())
    distinct=1+sum(len(set(int(u) for u in units[i,j] if u>0))
                   for i in range(len(layout.base_ids)) for j in range(len(layout.base_ids)))
    generic=.9*sum(a==action for a in predictions)/max(1,len(predictions))+.1/distinct
    return max(-8.,min(8.,log(max(1e-12,probability)/generic)))
