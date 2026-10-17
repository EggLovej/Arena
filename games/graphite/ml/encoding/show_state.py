"""Print what the network would see for one position.

Run from the game directory. The seed matches the map viewer, so the same
map can be open in both:

    .venv/bin/python -m ml show-state --seed 3
    .venv/bin/python -m ml show-state --seed 3 --seconds 20 --player blue
    .venv/bin/python -m ml show-state --seed 3 --seconds 20 --features v1

``--features`` picks the set of columns a network reads (``FEATURE_SETS``):
v2 (the default) or v1, what our networks up to 2026-10-08 read.
"""

import argparse

from game import Game
from game_types import BLUE, NEUTRAL, RED
from loading import load_ai
from main import play_headless

from ml.encoding.state_encoder import (
    BASE_FEATURES,
    FEATURE_SETS,
    GLOBAL_FEATURES,
    HORIZON_TICKS,
    PAIR_FEATURES,
    ROUTE_FEATURES,
    encode_state,
    layout_for,
)
from ml.encoding.forecast import build_timelines, ticks_until_arrival, units_to_capture
from ml.map_generator import generate_map


OWNER_NAMES = {RED: "red", BLUE: "blue", NEUTRAL: "-"}
SHORT_NAMES = {
    "is_mine": "mine",
    "is_enemy": "enemy",
    "is_neutral": "neutr",
    "production": "prod",
    "units_share": "share",
    "incoming_mine": "in_me",
    "incoming_enemy": "in_en",
    "mine_arrives_soon": "soon_me",
    "enemy_arrives_soon": "soon_en",
    "future_is_mine": "f_mine",
    "future_is_enemy": "f_enemy",
    "future_is_neutral": "f_neutr",
    "future_units": "f_units",
    "units_linear": "units",
    "production_linear": "prod",
    "incoming_mine_linear": "in_me",
    "incoming_enemy_linear": "in_en",
    "has_non_mine_neighbour": "front",
    "distance_to_enemy": "dist",
}
# The forecast columns of a set are shown as their own table, one column per horizon.
FORECAST_PREFIXES = ("in_", "future_")
OWNER_LETTERS = ("M", "E", "-")  # the one-hot mine / enemy / neutral, as one letter


def show(seed: int, seconds: float, player: int, asymmetry: float | None, feature_set: str = "v2") -> None:
    game = Game(generate_map(seed, asymmetry=asymmetry).definition)
    ai = load_ai("strategic_ai")
    play_headless(game, ai, ai, max_ticks=round(seconds / game.seconds_per_tick))

    state = game.view_for(player)
    encoded = encode_state(state)
    raw = {base.id: base for base in state.bases}
    chosen = FEATURE_SETS[feature_set]

    print(
        f"Seed {seed} after {game.time_seconds:.1f}s, seen by {OWNER_NAMES[player]}: "
        f"{len(state.bases)} bases, {len(state.edges)} routes, {len(state.fleets)} fleets in flight\n"
    )

    names = chosen["base_features"]
    now = [name for name in names if not name.startswith(FORECAST_PREFIXES)]
    print(
        f"BASE FEATURES, set {feature_set}: {len(names)} per base"
        + (", the forecast ones in the table after this" if len(now) < len(names) else "")
        + "  (left: the game's own values; right: what the network gets)"
    )
    headers = [SHORT_NAMES.get(name, name) for name in now]
    print(f"{'base':9}{'owner':>6}{'units':>7}{'prod':>6}  |" + "".join(f"{h:>8}" for h in headers))
    for base_id, row in zip(encoded.base_ids, encoded.base_features):
        base = raw[base_id]
        print(
            f"{base_id:9}{OWNER_NAMES[base.owner]:>6}{base.units:7.1f}{base.production_rate:6.2f}  |"
            + "".join(f"{row[BASE_FEATURES.index(name)]:8.2f}" for name in now)
        )
    if len(now) < len(names):
        show_forecast(encoded, raw, names)

    print("\nGLOBAL FEATURES")
    for name in chosen["global_features"]:
        print(f"  {name:36}{encoded.global_features[GLOBAL_FEATURES.index(name)]:6.2f}")

    my_bases = [base.id for base in state.bases if base.owner == player]
    source = my_bases[0] if my_bases else encoded.base_ids[0]
    row = encoded.base_ids.index(source)
    pair_columns = [PAIR_FEATURES.index(name) for name in chosen["pair_features"]]
    print(f"\nPAIR FEATURES from {source} to every other base")
    print(f"  {'to':9}" + "".join(f"{name:>20}" for name in chosen["pair_features"]))
    for column, base_id in enumerate(encoded.base_ids):
        if column != row:
            values = encoded.pair_features[row, column, pair_columns]
            print(f"  {base_id:9}" + "".join(f"{value:20.2f}" for value in values))
    if not chosen["route_features"]:
        print(f"\n(No route features in set {feature_set}.)")
        return
    route_columns = [ROUTE_FEATURES.index(name) for name in chosen["route_features"]]

    # Routes out of the same base, then routes into it: the first are its
    # attacks, the second what its neighbours would need to take it.
    layout = layout_for(state)
    timelines = build_timelines(state)
    print(f"\nROUTE FEATURES out of and into {source} (left: units the source's owner needs, worked out again)")
    print(
        f"  {'from':9}{'to':9}{'owner':>6}{'units':>7}{'prod':>6}{'needs':>7}  |"
        + "".join(f"{name:>20}" for name in chosen["route_features"])
    )
    for start, end in layout.routes:
        if row not in (start, end):
            continue
        source_base, target_base = raw[encoded.base_ids[start]], raw[encoded.base_ids[end]]
        travel = state.travel_time(source_base.id, target_base.id)
        needs = units_to_capture(
            target_base,
            timelines.get(target_base.id),
            ticks_until_arrival(travel, state.seconds_per_tick),
            source_base.owner,
            state.seconds_per_tick,
        )
        values = encoded.route_features[start, end, route_columns]
        print(
            f"  {source_base.id:9}{target_base.id:9}{OWNER_NAMES[target_base.owner]:>6}{target_base.units:7.1f}"
            f"{target_base.production_rate:6.2f}{max(needs, 0) if source_base.owner != NEUTRAL else '-':>7}  |"
            + "".join(f"{value:20.2f}" for value in values)
        )


def show_forecast(encoded, raw, names) -> None:
    """The set's forecast columns as a table: per moment, the owner (from its three columns) and the units."""

    moments = [f"in_{ticks}_ticks" for ticks in HORIZON_TICKS if f"in_{ticks}_ticks_units" in names] + ["future"]
    titles = [moment.replace("_", " ") for moment in moments[:-1]] + ["all landed"]
    future_units = "future_units_linear" if "future_units_linear" in names else "future_units"
    scale = "units / 100" if future_units == "future_units_linear" else "units on a log scale"
    print(
        "\nFORECAST if nobody sends anything new: owner (M mine, E enemy, - neutral) and units, "
        f"as the network gets them ({scale})"
    )
    print(f"{'base':9}{'owner':>6}{'units':>7}  |" + "".join(f"{title:>12}" for title in titles))
    for base_id, row in zip(encoded.base_ids, encoded.base_features):
        base = raw[base_id]
        cells = []
        for moment in moments:
            flags = [row[BASE_FEATURES.index(f"{moment}_is_{owner}")] for owner in ("mine", "enemy", "neutral")]
            units_name = future_units if moment == "future" else f"{moment}_units"
            cells.append(f"{OWNER_LETTERS[flags.index(1.0)]} {row[BASE_FEATURES.index(units_name)]:6.2f}")
        print(f"{base_id:9}{OWNER_NAMES[base.owner]:>6}{base.units:7.1f}  |" + "".join(f"{cell:>12}" for cell in cells))


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
    parser.add_argument("--features", choices=sorted(FEATURE_SETS), default="v2", help="Which set of columns to show")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show(args.seed, args.seconds, RED if args.player == "red" else BLUE, args.asymmetry, args.features)


if __name__ == "__main__":
    main()
