"""Print every move the network may choose from in one position.

Run from the game directory. The seed matches the map viewer:

    .venv/bin/python -m ml show-actions --seed 3
    .venv/bin/python -m ml show-actions --seed 3 --seconds 20 --player blue

With ``--network`` it shows what a network would do instead: how likely it
waits, how likely each route is, and the amounts it would send there, most
likely first. ``--network free`` is an untrained network with free amounts,
so it shows where the curves start; any checkpoint path works too (an option
network's options are added up by the units they send).

    .venv/bin/python -m ml show-actions --seed 3 --seconds 12 --network free
"""

import argparse

from game import Game
from game_types import BLUE, NEUTRAL, RED
from loading import load_ai
from main import play_headless

import torch

from ml.encoding.action_encoder import AMOUNT_OPTIONS, legal_actions
from ml.encoding.state_encoder import encode_state, layout_for
from ml.map_generator import generate_map


OWNER_NAMES = {RED: "red", BLUE: "blue", NEUTRAL: "neutral"}
COLUMN = max(len(name) for name in AMOUNT_OPTIONS) + 1


def show(seed: int, seconds: float, player: int, asymmetry: float | None, network: str | None = None) -> None:
    game = Game(generate_map(seed, asymmetry=asymmetry).definition)
    ai = load_ai("strategic_ai")
    play_headless(game, ai, ai, max_ticks=round(seconds / game.seconds_per_tick))

    state = game.view_for(player)
    layout = layout_for(state)
    legal = legal_actions(state, layout)
    bases = {base.id: base for base in state.bases}
    if network is not None:
        show_policy(state, legal, layout, bases, network, seed, game.time_seconds)
        return

    routes = int(legal.allowed.any(axis=-1).sum())
    print(
        f"Seed {seed} after {game.time_seconds:.1f}s, {OWNER_NAMES[player]} to move: "
        f"wait, or one of {int(legal.allowed.sum())} sends over {routes} routes\n"
    )
    print(
        f"{'from':9}{'has':>5}   {'to':9}{'owner':>8}{'units':>7}{'travel':>8}  |"
        + "".join(f"{name:>{COLUMN}}" for name in AMOUNT_OPTIONS)
    )
    for source, source_id in enumerate(legal.base_ids):
        for target, travel_seconds in layout.neighbours[source]:
            if not legal.allowed[source, target].any():
                continue
            target_base = bases[legal.base_ids[target]]
            cells = "".join(
                f"{units:{COLUMN}d}" if allowed else f"{'-':>{COLUMN}}"
                for allowed, units in zip(legal.allowed[source, target], legal.units[source, target])
            )
            print(
                f"{source_id:9}{int(bases[source_id].units):5d}   {target_base.id:9}"
                f"{OWNER_NAMES[target_base.owner]:>8}{target_base.units:7.1f}{travel_seconds:7.1f}s  |{cells}"
            )


def show_policy(state, legal, layout, bases, network: str, seed: int, seconds: float) -> None:
    """What a network would do here: P(wait), P(route), and the likeliest amounts per route."""

    from ml.models import amounts, make_input
    from ml.models.checkpoint import init_model

    model = init_model("random", seed=0, amounts="free").model if network == "free" else init_model(network).model
    model.eval()
    with torch.no_grad():
        output = model(make_input([encode_state(state, layout)], [legal]))
    route_chance = torch.softmax(output.route_logits[0], dim=0)
    count = len(legal.base_ids)
    kind = "free amounts" if model.free_amounts else f"{len(AMOUNT_OPTIONS)} amount options, added up by units"
    print(
        f"Seed {seed} after {seconds:.1f}s, {OWNER_NAMES[state.player_id]} to move. Network: {network} ({kind})\n"
        f"P(wait) = {route_chance[0]:.1%}\n"
    )
    print(f"{'from':9}{'has':>5}   {'to':9}{'owner':>8}{'units':>7}{'needs':>6}  |{'P(route)':>9}  | likeliest amounts (units probability)")
    rows = []
    for source, source_id in enumerate(legal.base_ids):
        for target, _ in layout.neighbours[source]:
            if not legal.allowed[source, target].any():
                continue
            if model.free_amounts:
                most = output.most[0, source, target]
                chances = amounts.on_grid(
                    output.amount_curves[0, source, target], most, output.needed[0, source, target], int(most)
                ).tolist()
                by_units = {units: chance for units, chance in enumerate(chances, start=1)}
            else:
                chances = torch.softmax(output.amount_logits[0, source, target], dim=0).tolist()
                by_units = {}
                for units, allowed, chance in zip(legal.units[source, target], legal.allowed[source, target], chances):
                    if allowed:
                        by_units[int(units)] = by_units.get(int(units), 0.0) + chance
            rows.append((route_chance[1 + source * count + target].item(), source, target, by_units))
    for chance, source, target, by_units in sorted(rows, reverse=True):
        source_id, target_base = legal.base_ids[source], bases[legal.base_ids[target]]
        best = sorted(by_units.items(), key=lambda item: -item[1])[:5]
        needs = int(legal.needed[source, target])
        print(
            f"{source_id:9}{int(bases[source_id].units):5d}   {target_base.id:9}{OWNER_NAMES[target_base.owner]:>8}"
            f"{target_base.units:7.1f}{needs if needs else '-':>6}  |{chance:9.1%}  | "
            + "  ".join(f"{units:3d} {share:4.0%}" for units, share in best if share >= 0.01)
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Let strategic_ai play both sides this long first (default: the starting position)",
    )
    parser.add_argument("--player", choices=("red", "blue"), default="red")
    parser.add_argument("--asymmetry", type=float, help="Fix the asymmetry level, as in the viewer")
    parser.add_argument("--network", help='Show what this network would do: a checkpoint path, or "free" (untrained, free amounts)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show(args.seed, args.seconds, RED if args.player == "red" else BLUE, args.asymmetry, args.network)


if __name__ == "__main__":
    main()
