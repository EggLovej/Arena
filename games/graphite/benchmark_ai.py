"""Reproducible all-seat AI tournaments; timeouts are NOT counted as wins."""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from game import Game
from loading import available_maps, load_ai
from generated_maps import load_case


def outcome_for(game, seat):
    if game.winner() == seat:
        return 'win'
    if seat not in game.living_players():
        return 'loss'
    return 'timeout'


def match(job):
    name, candidate, opponents, seat, seconds, dt = job
    definition = load_case(name)
    game=Game(definition,dt)
    candidate_ai=load_ai(candidate)
    enemies=[load_ai(o) for o in opponents]
    controllers={p:candidate_ai if p==seat else enemies.pop(0) for p in game.players}
    durations=[]
    for tick in range(int(seconds/dt)):
        actions={}
        for p in game.players:
            start=time.perf_counter()
            actions[p]=controllers[p](game.view_for(p))
            if p==seat: durations.append(time.perf_counter()-start)
        errors=game.step(actions)
        if errors: raise RuntimeError(errors)
        if game.is_finished(): break
    winner=game.winner()
    strength={p:sum(b.units+24*b.production_rate for b in game.bases.values() if b.owner==p)+sum(f.units for f in game.fleets if f.owner==p) for p in game.players}
    return dict(map=name, candidate=candidate, opponents=opponents, seat=seat,
                outcome=outcome_for(game,seat),
                seconds=game.time_seconds, strength=strength, living_players=list(game.living_players()),
                mean_ms=1000*sum(durations)/len(durations), max_ms=1000*max(durations))


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--candidate',default='atlas_ai')
    a.add_argument('--opponents',nargs='+',default=['aggressive_ai','careful_ai','strategic_ai'])
    a.add_argument('--maps',nargs='+')
    a.add_argument('--generated',type=int,default=0)
    a.add_argument('--seed',type=int,default=10000)
    a.add_argument('--mixed',action='store_true')
    a.add_argument('--seconds',type=float,default=180)
    a.add_argument('--dt',type=float,default=.1)
    a.add_argument('--workers',type=int,default=4)
    a.add_argument('--output',default='benchmark_results.json')
    args=a.parse_args()
    if args.seconds <= 0 or args.dt <= 0 or args.workers < 1 or args.generated < 0:
        a.error('seconds, dt, workers must be positive; generated must be nonnegative')
    if int(args.seconds / args.dt) < 1:
        a.error('seconds must allow at least one tick')
    jobs=[]
    names=args.maps or ([f'generated:{seed}:{p}' for seed in range(args.seed,args.seed+args.generated) for p in (2,3)] if args.generated else available_maps())
    from itertools import product
    for name in names:
        players=Game(load_case(name)).players
        rosters=list(product(args.opponents,repeat=len(players)-1)) if args.mixed else [(enemy,)*(len(players)-1) for enemy in args.opponents]
        for roster in rosters:
            for seat in players:
                jobs.append((name,args.candidate,list(roster),seat,args.seconds,args.dt))
    results=[]
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(match,jobs):
            results.append(r)
            output.write_text(json.dumps(results,indent=2)+'\n')
            print(r['map'],r['opponents'],r['seat'],r['outcome'],r['seconds'],flush=True)
    Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
    print(dict(Counter(r['outcome'] for r in results)),flush=True)
    print('average decision ms',sum(r['mean_ms'] for r in results)/len(results))

if __name__=='__main__': main()
