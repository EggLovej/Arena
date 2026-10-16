"""Where does a network fail to copy strategic_ai?

Imitation reports one number: how often the network's most likely move is
the teacher's. This takes that number apart. strategic_ai is played on fresh
maps against itself and the two simpler AIs, and every decision of its seat
is sorted twice:

- by which part of strategic_ai made the move: defence (reinforce a base
  that is under attack), attack (one base takes a neighbour), pressure
  (several bases send everything at one target), reinforcement (move
  surplus towards the border), or wait
- by what the network did instead: the same move; the same route and the
  same number of units under another amount option; the same route with
  the neighbouring amount option; the same route with an amount further
  off; another target from the same base; another base; or waiting

A route the network picks that strategic_ai itself uses within the next
second (``--soon-seconds``) is counted apart: it is the teacher's own
move, only in another order.

From the game directory:

    .venv/bin/python -m ml mistakes ml/runs/inputs_before_seed0/latest.pt
"""

import argparse
from collections import Counter
from unittest import mock

import numpy as np
import torch

import ais.strategic_ai as strategic
from game import Game
from game_types import BLUE, RED, GameView
from loading import load_ai

from ml.encoding.action_encoder import encode_action, legal_actions
from ml.encoding.forecast import build_timelines
from ml.encoding.state_encoder import encode_state, layout_for
from ml.evaluation.evaluate import FIRST_EVALUATION_SEED, MATCH_SECONDS
from ml.map_generator import generate_map
from ml.models import make_input
from ml.models.checkpoint import load_checkpoint


OPPONENTS = ("strategic_ai", "careful_ai", "aggressive_ai")
# Maps no trainer and no other check has used.
FIRST_SEED = FIRST_EVALUATION_SEED + 5_000
PARTS = ("defence", "attack", "pressure", "reinforcement", "wait")
OUTCOMES = (
    "same move",
    "same units, other option",
    "neighbouring amount",
    "amount further off",
    "other target, same base",
    "other base",
    "teacher's route, soon after",
    "waits",
)


def part_of_strategic_ai(state: GameView) -> str:
    """Which branch of strategic_ai.choose_action produces its move here."""

    bases = {base.id: base for base in state.bases}
    graph = strategic._build_graph(state)
    if strategic._best_defense(state, bases, graph) is not None:
        return "defence"
    # _best_attack falls back to a pressure attack when no single base can attack.
    with mock.patch.object(strategic, "_best_pressure_attack", return_value=None):
        if strategic._best_attack(state, bases, graph) is not None:
            return "attack"
    if strategic._best_pressure_attack(state, bases, graph) is not None:
        return "pressure"
    if strategic._best_reinforcement(state, bases, graph) is not None:
        return "reinforcement"
    return "wait"


def play_and_record(games: int, first_seed: int, soon_seconds: float):
    """Teacher decisions from fresh games: (position, legal choices, teacher's choice, part, soon).

    ``soon`` is the set of routes, as (source row, target row), that the
    teacher sends along within ``soon_seconds`` after this decision.
    """

    teacher = load_ai("strategic_ai")
    opponents = {name: load_ai(name) for name in OPPONENTS}
    decisions = []
    for number in range(games):
        game = Game(generate_map(first_seed + number).definition)
        opponent = opponents[OPPONENTS[number % len(OPPONENTS)]]
        seat = RED if number % 2 == 0 else BLUE
        other = BLUE if seat == RED else RED
        soon_ticks = round(soon_seconds / game.seconds_per_tick)
        layout = None
        recorded = []  # (tick, decision) for this game
        sends = []  # (tick, source row, target row) of every teacher send in this game
        while not game.is_finished() and game.time_seconds < MATCH_SECONDS - 1e-9:
            state = game.view_for(seat)
            move = teacher(state)
            layout = layout or layout_for(state)
            timelines = build_timelines(state)
            legal = legal_actions(state, layout, timelines)
            if legal.allowed.any():
                choice = encode_action(move, legal)
                part = part_of_strategic_ai(state) if move is not None else "wait"
                recorded.append((game.tick, [encode_state(state, layout, timelines), legal, choice, part, set()]))
            if move is not None:
                sends.append((game.tick, layout.index[move.source_id], layout.index[move.target_id]))
            game.step({seat: move, other: opponent(game.view_for(other))})
        for tick, decision in recorded:
            decision[4].update((source, target) for when, source, target in sends if tick < when <= tick + soon_ticks)
            decisions.append(tuple(decision))
    return decisions


def network_choices(model, decisions, batch_size: int = 256):
    """The network's single most likely move in each position, as imitation measures it."""

    choices = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(decisions), batch_size):
            chunk = decisions[start : start + batch_size]
            inputs = make_input([entry[0] for entry in chunk], [entry[1] for entry in chunk])
            output = model(inputs)
            bases = inputs.base_mask.shape[1]
            route = output.route_logits.argmax(dim=1)
            per_route = output.amount_logits.flatten(start_dim=1, end_dim=2)
            amount = per_route[torch.arange(len(chunk)), (route - 1).clamp(min=0)].argmax(dim=1)
            for route_index, amount_index in zip(route.tolist(), amount.tolist()):
                if route_index == 0:
                    choices.append(None)
                else:
                    choices.append(((route_index - 1) // bases, (route_index - 1) % bases, amount_index))
    return choices


def outcome(teacher, network, legal, soon) -> str:
    """What the network did, for a decision where the teacher sends."""

    if network is None:
        return "waits"
    if (network[0], network[1]) != (teacher.source, teacher.target) and (network[0], network[1]) in soon:
        return "teacher's route, soon after"
    if network[0] != teacher.source:
        return "other base"
    if network[1] != teacher.target:
        return "other target, same base"
    if network[2] == teacher.amount:
        return "same move"
    units = legal.units[teacher.source, teacher.target]
    if units[network[2]] == units[teacher.amount]:
        return "same units, other option"
    # Amount options in order of the units they send, so "neighbouring" means next larger or smaller.
    offered = sorted({int(value) for value in units if value > 0})
    gap = abs(offered.index(int(units[network[2]])) - offered.index(int(units[teacher.amount])))
    return "neighbouring amount" if gap == 1 else "amount further off"


def report(name: str, decisions, choices) -> str:
    table = Counter()
    for (_, legal, teacher, part, soon), network in zip(decisions, choices):
        if teacher is None:
            if network is None:
                table[part, "same move"] += 1
            else:
                table[part, "soon" if (network[0], network[1]) in soon else "sends"] += 1
        else:
            table[part, outcome(teacher, network, legal, soon)] += 1

    sends = sum(count for (part, _), count in table.items() if part != "wait")
    lines = [f"{name}: {len(decisions):,} decisions, {sends:,} of them sends", ""]
    width = max(len(text) for text in OUTCOMES)
    lines.append(f"{'':{width}}" + "".join(f"{part:>15}" for part in PARTS[:-1]) + f"{'all sends':>15}")
    totals = {part: sum(table[part, text] for text in OUTCOMES) for part in PARTS[:-1]}
    lines.append(
        f"{'share of sends':{width}}"
        + "".join(f"{totals[part] / max(1, sends):15.1%}" for part in PARTS[:-1])
        + f"{1:15.1%}"
    )
    for text in OUTCOMES:
        row = [table[part, text] / max(1, totals[part]) for part in PARTS[:-1]]
        overall = sum(table[part, text] for part in PARTS[:-1]) / max(1, sends)
        lines.append(f"{text:{width}}" + "".join(f"{value:15.1%}" for value in row) + f"{overall:15.1%}")
    waits = table["wait", "same move"] + table["wait", "sends"]
    lines.append("")
    waits += table["wait", "soon"]
    lines.append(
        f"Teacher waits: {waits:,} decisions; the network also waits in {table['wait', 'same move'] / max(1, waits):.1%}, "
        f"sends along a route the teacher uses soon after in {table['wait', 'soon'] / max(1, waits):.1%}"
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", nargs="+", help="Networks trained to copy strategic_ai")
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--first-seed", type=int, default=FIRST_SEED)
    parser.add_argument("--soon-seconds", type=float, default=1.0, help="How far ahead a teacher move still counts as 'soon after'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    decisions = play_and_record(args.games, args.first_seed, args.soon_seconds)
    for path in args.checkpoints:
        model = load_checkpoint(path).model
        if model.free_amounts:
            raise SystemExit(f"{path} has free amounts; this report compares amount options")
        print(report(path, decisions, network_choices(model, decisions)))
        print()


if __name__ == "__main__":
    main()
