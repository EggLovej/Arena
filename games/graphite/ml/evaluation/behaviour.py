"""How networks play, rather than how well: where their sends go, and how decided they are.

Each checkpoint plays the same fresh maps against itself, under the send
limit its run trained under if it had one (``--min-send-gap``), and its
recorded decisions are measured:

- how often it sends (per second, and how often it waits when it may send),
  how many units a send carries, and how long its games last,
- sends per game and seat, and the share going to its own bases (moving
  units around) rather than to neutral or enemy bases,
- units away from the front: how far, on average, its units are from the
  nearest enemy base (0: all next to it or attacking; see
  ``ml.training.selfplay.front_potential``),
- how decided its amounts are: at the positions where it sent something,
  the uncertainty of its amount choice as a share of picking evenly among
  the legal amounts (100%: a coin toss), and the probability of its
  favourite amount. For a network with free amounts, the legal amounts are
  every whole number of units from 1 to everything at the source,
- how often it takes the tactical amounts "hold" and "surplus" when they are
  on offer (with free amounts: sends exactly that many units).

    .venv/bin/python -m ml behaviour ml/runs/ab_plain/round_0400.pt ml/runs/large_selfplay/round_0150.pt
"""

import argparse

import numpy as np
import torch

from ml.encoding.action_encoder import HOLD, SURPLUS, option_units
from ml.evaluation.saved_games import short_name
from ml.map_generator import generate_map
from ml.models import amounts
from ml.models.checkpoint import load_checkpoint
from ml.models.policy import _amount_logits_for, _free_amounts_for
from ml.play.rollout import WAIT, GameRecord, Matchup, RolloutPool, Seat, Trajectory
from ml.training.dataset import DecisionSet
from ml.training.runlog import min_send_gap
from ml.training.selfplay import behaviour as send_behaviour

# Fresh maps, past those used by training, evaluation, comparisons and saved checks.
FIRST_MAP = 990_000


def measure(path: str, games: int = 48, workers: int = 6, pool: RolloutPool | None = None) -> dict[str, float]:
    model = load_checkpoint(path).model
    maps = [generate_map(FIRST_MAP + number).definition for number in range(games)]
    gap = min_send_gap(path)
    network = Seat("network", True, min_send_gap=gap)
    matchups = [Matchup(definition, network, network) for definition in maps]
    if pool is None:
        with RolloutPool(workers) as own_pool:
            records = own_pool.play(matchups, {"network": model}, seed=0)
    else:
        records = pool.play(matchups, {"network": model}, seed=0)
    trajectories = [trajectory for record in records for trajectory in record.trajectories]
    return {
        "send gap, ticks": float(gap),
        **send_behaviour(records),
        **_pace(records),
        **_amounts(model, trajectories),
        **_tactical(trajectories),
    }


def _pace(records: list[GameRecord]) -> dict[str, float]:
    """How often it sends and how much, and how long its games last (both seats are the network)."""

    units = [action[-1] for record in records for action in record.actions]
    seat_seconds = sum(record.ticks * 0.1 * len(record.trajectories) for record in records)
    return {
        "sends per second": len(units) / seat_seconds,
        "units per send": float(np.mean(units)) if units else float("nan"),
        "game length, seconds": float(np.mean([record.ticks * 0.1 for record in records])),
        "decided games": float(np.mean([record.winner is not None for record in records])),
    }


def _amounts(model, trajectories: list[Trajectory]) -> dict[str, float]:
    """How decided the amount is at the positions where the network sent something."""

    # A sample of each seat's decisions is plenty, and keeps memory small.
    decisions, rng = DecisionSet(model.free_amounts), np.random.default_rng(0)
    for trajectory in trajectories:
        decisions.add(trajectory, None, 300, rng)
    uncertainty, favourite = [], []
    model.eval()
    with torch.no_grad():
        for batch in decisions.batches(512):
            sent = batch.route > 0
            if not sent.any():
                continue
            output = model(batch.inputs)
            if model.free_amounts:
                curves, most, needed = (part[sent] for part in _free_amounts_for(output, batch.route))
                size = int(min(most.max().item(), amounts.GRID_LIMIT))
                chances = amounts.on_grid(curves, most, needed, size)
                legal = most.clamp(max=size)
            else:
                logits = _amount_logits_for(output, batch.route)[sent]
                legal = (logits > -1e8).sum(dim=1)
                chances = torch.softmax(logits, dim=1)
            entropy = -(chances * torch.log(chances.clamp(min=1e-12))).sum(dim=1)
            several = legal > 1
            uncertainty += (entropy[several] / torch.log(legal[several].float())).tolist()
            favourite += chances.max(dim=1).values[several].tolist()
    return {
        "amount uncertainty, share of an even pick": float(np.mean(uncertainty)),
        "favourite amount's probability": float(np.mean(favourite)),
    }


def _tactical(trajectories: list[Trajectory]) -> dict[str, float]:
    offered = {HOLD: [0, 0], SURPLUS: [0, 0]}  # sends where it was on offer, and of those, taken
    for trajectory in trajectories:
        sends = np.flatnonzero(trajectory.source != WAIT)
        if not len(sends):
            continue
        allowed = trajectory.allowed(sends)
        units = option_units(trajectory.limit_table(sends)) if trajectory.free_amounts else None
        for row, number in enumerate(sends):
            source, target = trajectory.source[number], trajectory.target[number]
            route = allowed[row, source, target]
            for option in offered:
                if route[option]:
                    offered[option][0] += 1
                    if units is None:
                        offered[option][1] += int(trajectory.amount[number]) == option
                    else:
                        offered[option][1] += int(trajectory.units[number]) == units[row, source, target, option]
    return {
        f"{name} taken when on offer": taken / shown if shown else float("nan")
        for name, (shown, taken) in (("hold", offered[HOLD]), ("surplus", offered[SURPLUS]))
    }


def table(results: dict[str, dict[str, float]]) -> str:
    rows = [
        ("send gap (ticks, 0: no limit)", "send gap, ticks", "{:.0f}"),
        ("sends per game and seat", "sends/per game and seat", "{:.0f}"),
        ("sends per second", "sends per second", "{:.1f}"),
        ("waits when it may send", "behaviour/waits when it may send", "{:.0%}"),
        ("units per send", "units per send", "{:.1f}"),
        ("sends to own bases", "behaviour/sends to own bases", "{:.0%}"),
        ("units away from the front", "behaviour/units away from the front", "{:.0%}"),
        ("game length, seconds", "game length, seconds", "{:.0f}"),
        ("decided games", "decided games", "{:.0%}"),
        ("amount uncertainty (100%: even)", "amount uncertainty, share of an even pick", "{:.0%}"),
        ("favourite amount's probability", "favourite amount's probability", "{:.0%}"),
        ("hold, taken when on offer", "hold taken when on offer", "{:.1%}"),
        ("surplus, taken when on offer", "surplus taken when on offer", "{:.1%}"),
    ]
    names = [short_name(path) for path in results]
    label = max(len(row[0]) for row in rows)
    width = max(14, *(len(name) for name in names))
    lines = [" " * label + "".join(f"  {name:>{width}}" for name in names)]
    for title, key, form in rows:
        cells = []
        for values in results.values():
            value = values.get(key, float("nan"))
            cells.append("-" if value != value else form.format(value))
        lines.append(f"{title:<{label}}" + "".join(f"  {cell:>{width}}" for cell in cells))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--games", type=int, default=48, help="Self-play games per checkpoint (default 48)")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    results = {}
    with RolloutPool(args.workers) as pool:
        for path in args.checkpoints:
            results[path] = measure(path, args.games, pool=pool)
            print(f"measured {path}", flush=True)
    print("\n" + table(results), flush=True)


if __name__ == "__main__":
    main()
