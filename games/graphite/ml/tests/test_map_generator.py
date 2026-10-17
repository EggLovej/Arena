import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from game import Game
from game_types import BLUE, NEUTRAL, RED, BaseDefinition, EdgeDefinition, MapDefinition
from loading import load_ai, validate_map
from main import play_headless

from ml.map_generator import DEFAULT_PARAMS, MapGenParams, generate_map, map_to_json, save_map


SEEDS = range(300)


def routes_by_base(definition):
    routes = {base.id: {} for base in definition.bases}
    for edge in definition.edges:
        routes[edge.a][edge.b] = edge.travel_time
        routes[edge.b][edge.a] = edge.travel_time
    return routes


def is_mirrored(definition):
    """True when swapping left and right turns the map into itself with colours swapped."""

    by_position = {(base.x, base.y): base for base in definition.bases}
    twin = {}
    for base in definition.bases:
        other = by_position.get((definition.width - base.x, base.y))
        if other is None:
            return False
        if other.owner != -base.owner:  # RED is -1, BLUE is 1, NEUTRAL is 0
            return False
        if (other.units, other.production_rate) != (base.units, base.production_rate):
            return False
        twin[base.id] = other.id

    routes = routes_by_base(definition)
    return all(
        routes[twin[a]].get(twin[b]) == travel_time
        for a, neighbours in routes.items()
        for b, travel_time in neighbours.items()
    )


def crosses(definition, first, second):
    if {first.a, first.b} & {second.a, second.b}:
        return False
    at = {base.id: (base.x, base.y) for base in definition.bases}

    def side(a, b, point):
        return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])

    a, b, c, d = at[first.a], at[first.b], at[second.a], at[second.b]
    return side(c, d, a) * side(c, d, b) < 0 and side(a, b, c) * side(a, b, d) < 0


class MapGeneratorTests(unittest.TestCase):
    def test_same_seed_gives_same_map(self):
        for seed in (0, 1, 99):
            self.assertEqual(generate_map(seed), generate_map(seed))
            self.assertEqual(generate_map(seed, asymmetry=0.4), generate_map(seed, asymmetry=0.4))

    def test_different_seeds_give_different_maps(self):
        maps = {map_to_json(generate_map(seed).definition) for seed in range(50)}
        self.assertEqual(len(maps), 50)

    def test_every_map_is_a_valid_two_player_map(self):
        for seed in SEEDS:
            definition = generate_map(seed).definition
            validate_map(definition)
            self.assertEqual(Game(definition).players, (RED, BLUE), seed)
            self.assertTrue(any(base.owner == NEUTRAL for base in definition.bases), seed)

    def test_values_stay_inside_the_limits(self):
        params = DEFAULT_PARAMS
        for seed in SEEDS:
            for asymmetry in (None, 1.0):
                definition = generate_map(seed, asymmetry=asymmetry).definition
                for base in definition.bases:
                    self.assertGreaterEqual(base.production_rate, params.min_production)
                    self.assertLessEqual(base.production_rate, params.max_production)
                    if base.owner == NEUTRAL:
                        low, high = params.min_neutral_units, params.max_neutral_units
                    else:
                        low, high = params.min_starting_units, params.max_starting_units
                    self.assertGreaterEqual(base.units, low)
                    self.assertLessEqual(base.units, high)
                    self.assertEqual(base.units, int(base.units))
                for edge in definition.edges:
                    self.assertGreaterEqual(edge.travel_time, params.min_travel_time)
                    self.assertLessEqual(edge.travel_time, params.max_travel_time)

    def test_size_and_routes_stay_inside_the_limits(self):
        params = DEFAULT_PARAMS
        smallest = 2 * params.min_bases_per_side
        largest = 2 * params.max_bases_per_side + params.max_centre_bases
        for seed in SEEDS:
            definition = generate_map(seed).definition
            self.assertGreaterEqual(len(definition.bases), smallest)
            self.assertLessEqual(len(definition.bases), largest)
            for neighbours in routes_by_base(definition).values():
                self.assertLessEqual(len(neighbours), params.max_routes_at_one_base)
            for base in definition.bases:
                self.assertGreaterEqual(base.x, params.field_margin_x)
                self.assertLessEqual(base.x, params.width - params.field_margin_x)
                self.assertGreaterEqual(base.y, params.field_top)
                self.assertLessEqual(base.y, params.field_bottom)

    def test_asymmetry_zero_is_an_exact_mirror(self):
        for seed in SEEDS:
            definition = generate_map(seed, asymmetry=0.0).definition
            self.assertTrue(is_mirrored(definition), seed)

    def test_asymmetry_breaks_the_mirror(self):
        broken = sum(
            not is_mirrored(generate_map(seed, asymmetry=0.3).definition)
            for seed in range(50)
        )
        self.assertEqual(broken, 50)

    def test_usual_mix_contains_both_kinds(self):
        levels = [generate_map(seed).asymmetry for seed in SEEDS]
        mirrored = sum(level == 0 for level in levels)
        self.assertGreater(mirrored, 0.35 * len(levels))
        self.assertLess(mirrored, 0.65 * len(levels))
        self.assertLessEqual(max(levels), DEFAULT_PARAMS.max_asymmetry)

    def test_routes_never_cross_or_crowd(self):
        for seed in range(100):
            for asymmetry in (0.0, 1.0):
                definition = generate_map(seed, asymmetry=asymmetry).definition
                edges = definition.edges
                for index, first in enumerate(edges):
                    for second in edges[index + 1 :]:
                        self.assertFalse(crosses(definition, first, second), (seed, asymmetry))
                bases = definition.bases
                for index, first in enumerate(bases):
                    for second in bases[index + 1 :]:
                        distance = math.dist((first.x, first.y), (second.x, second.y))
                        self.assertGreaterEqual(distance, DEFAULT_PARAMS.min_base_distance)

    def test_each_player_can_afford_a_first_capture(self):
        for seed in SEEDS:
            for asymmetry in (0.0, 1.0):
                definition = generate_map(seed, asymmetry=asymmetry).definition
                owners = {base.id: base for base in definition.bases}
                for player in (RED, BLUE):
                    waits = []
                    for source_id, neighbours in routes_by_base(definition).items():
                        source = owners[source_id]
                        if source.owner != player:
                            continue
                        for target_id in neighbours:
                            target = owners[target_id]
                            if target.owner != NEUTRAL:
                                continue
                            shortfall = target.units + 1 - source.units
                            if shortfall <= 0:
                                waits.append(0.0)
                            elif source.production_rate > 0:
                                waits.append(shortfall / source.production_rate)
                    self.assertTrue(waits, (seed, asymmetry))
                    self.assertLessEqual(min(waits), DEFAULT_PARAMS.max_opening_wait_seconds)

    def test_names_and_order_do_not_reveal_roles(self):
        first_owner = {generate_map(seed).definition.bases[0].owner for seed in range(100)}
        self.assertEqual(first_owner, {RED, NEUTRAL, BLUE})

    def test_custom_limits_are_respected(self):
        params = replace(
            DEFAULT_PARAMS,
            max_production=1.0,
            max_neutral_units=10,
            max_starting_units=20,
            min_bases_per_side=3,
            max_bases_per_side=4,
            max_centre_bases=0,
        )
        for seed in range(100):
            definition = generate_map(seed, params).definition
            self.assertIn(len(definition.bases), (6, 8))
            for base in definition.bases:
                self.assertLessEqual(base.production_rate, 1.0)
                self.assertLessEqual(base.units, 10 if base.owner == NEUTRAL else 20)

    def test_inverted_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            MapGenParams(min_production=5, max_production=1)

    def test_saved_map_loads_back_identically(self):
        definition = generate_map(5).definition
        with tempfile.TemporaryDirectory() as directory:
            path = save_map(definition, Path(directory))
            self.assertEqual(path.name, "generated_5.json")
            data = json.loads(path.read_text())

        # Rebuilt the same way loading.load_map does.
        loaded = MapDefinition(
            name=data["name"],
            width=data["width"],
            height=data["height"],
            bases=tuple(BaseDefinition(**base) for base in data["bases"]),
            edges=tuple(EdgeDefinition(**edge) for edge in data["edges"]),
        )
        self.assertEqual(loaded, definition)

    def test_scripted_ais_can_play_generated_maps(self):
        strategic = load_ai("strategic_ai")
        aggressive = load_ai("aggressive_ai")
        decided = 0
        for seed in range(20):
            game = Game(generate_map(seed).definition)
            decided += play_headless(game, strategic, aggressive, max_ticks=1800) is not None
            self.assertGreater(game.tick, 0)
        self.assertGreater(decided, 10)


if __name__ == "__main__":
    unittest.main()
