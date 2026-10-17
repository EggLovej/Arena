"""Print what a model makes of one position.

Run from the game directory. The seed matches the map viewer:

    .venv/bin/python -m ml show-model --seed 3
    .venv/bin/python -m ml show-model --seed 3 --seconds 20 --init ml/runs/demo/random.pt
"""

import argparse

import torch

from game import Game
from game_types import BLUE, RED
from loading import load_ai
from main import play_headless

from ml.encoding.action_encoder import AMOUNT_OPTIONS, legal_actions
from ml.encoding.state_encoder import encode_state
from ml.map_generator import generate_map
from ml.models import DEFAULT_MODEL, MODELS, make_input
from ml.models.checkpoint import init_model
from ml.models.policy import entropy


def show(init: str, model_name: str, seed: int, seconds: float, player: int, weights_seed: int) -> None:
    checkpoint = init_model(init, model_name, weights_seed)
    model = checkpoint.model
    if model.free_amounts:
        raise SystemExit(f"{init} has free amounts; see it with: ml show-actions --network {init}")

    print(f"Model '{checkpoint.model_name}': {sum(p.numel() for p in model.parameters()):,} weights")
    for name, part in model.named_children():
        print(f"  {name:16}{sum(p.numel() for p in part.parameters()):>9,}")

    game = Game(generate_map(seed).definition)
    ai = load_ai("strategic_ai")
    play_headless(game, ai, ai, max_ticks=round(seconds / game.seconds_per_tick))
    state = game.view_for(player)
    legal = legal_actions(state)

    with torch.no_grad():
        output = model(make_input([encode_state(state)], [legal]))
    bases = len(legal.base_ids)
    route_probability = torch.softmax(output.route_logits[0], dim=0)
    amount_probability = torch.softmax(output.amount_logits[0], dim=-1)

    routes = int(legal.allowed.any(axis=-1).sum())
    print(f"\nSeed {seed} after {game.time_seconds:.1f}s: wait or {routes} routes to choose from")
    print(f"  value (-1 losing, +1 winning)   {output.value[0]:+.3f}")
    print(f"  undecidedness (entropy)         {entropy(output)[0]:.2f}")
    print(f"  chance of waiting               {route_probability[0]:.1%}")

    print("\nMost likely routes, and how the amount would be chosen on each:")
    print(f"  {'route':22}{'chance':>8}  |" + "".join(f"{name:>12}" for name in AMOUNT_OPTIONS))
    for column in route_probability[1:].argsort(descending=True)[:8].tolist():
        source, target = divmod(column, bases)
        if not legal.allowed[source, target].any():
            break
        label = f"{legal.base_ids[source]} -> {legal.base_ids[target]}"
        amounts = "".join(
            f"{probability:12.1%}" if allowed else f"{'-':>12}"
            for allowed, probability in zip(legal.allowed[source, target], amount_probability[source, target])
        )
        print(f"  {label:22}{route_probability[1 + column]:8.1%}  |{amounts}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--init", default="random", help='"random" or the path of a checkpoint')
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS), help="Used with --init random")
    parser.add_argument("--seed", type=int, default=3, help="Map seed")
    parser.add_argument("--seconds", type=float, default=0.0, help="Let strategic_ai play this long first")
    parser.add_argument("--player", choices=("red", "blue"), default="red")
    parser.add_argument("--weights-seed", type=int, default=0, help="Used with --init random")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    player = RED if args.player == "red" else BLUE
    show(args.init, args.model, args.seed, args.seconds, player, args.weights_seed)


if __name__ == "__main__":
    main()
