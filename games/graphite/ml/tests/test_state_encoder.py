import copy
import random
import unittest
from dataclasses import replace

import numpy as np

from game import Game
from game_types import BLUE, NEUTRAL, RED, EdgeView
from loading import available_maps, load_ai, load_map

from ml.encoding.action_encoder import JUST_ENOUGH, legal_actions

from ml.encoding.state_encoder import (
    BASE_FEATURES,
    FEATURE_SETS,
    GLOBAL_FEATURES,
    HORIZON_TICKS,
    PAIR_FEATURES,
    PATH_SECONDS_REFERENCE,
    ROUTE_FEATURES,
    TRAVEL_SECONDS_REFERENCE,
    build_layout,
    encode_state,
    stack_states,
)
from ml.map_generator import generate_map


_UNITS_SCALE = 1 / np.log1p(250.0)


def column(name):
    return BASE_FEATURES.index(name)


def played_game(definition, ticks=150):
    """A position from the middle of a match, with fleets in flight."""

    game = Game(definition)
    strategic = load_ai("strategic_ai")
    aggressive = load_ai("aggressive_ai")
    for _ in range(ticks):
        if game.is_finished():
            break
        game.step({RED: strategic(game.view_for(RED)), BLUE: aggressive(game.view_for(BLUE))})
    return game


def shortest_seconds(state):
    """Shortest travel time between every two bases, worked out here independently of the encoder."""

    ids = [base.id for base in state.bases]
    time = {a: {b: 0.0 if a == b else float("inf") for b in ids} for a in ids}
    for edge in state.edges:
        time[edge.a][edge.b] = time[edge.b][edge.a] = min(time[edge.a][edge.b], edge.travel_time)
    for stop in ids:
        for a in ids:
            for b in ids:
                time[a][b] = min(time[a][b], time[a][stop] + time[stop][b])
    return time


def two_player_definitions():
    definitions = [generate_map(seed).definition for seed in range(8)]
    for name in available_maps():
        definition = load_map(name)
        if Game(definition).players == (RED, BLUE):
            definitions.append(definition)
    return definitions


def renamed_and_shuffled(state, seed):
    """The same position with new base names and every list in a new order."""

    rng = random.Random(seed)
    new_id = {base.id: f"x{rng.getrandbits(32):08x}" for base in state.bases}
    bases = [replace(base, id=new_id[base.id], x=rng.randrange(1000), y=rng.randrange(800)) for base in state.bases]
    edges = []
    for edge in state.edges:
        a, b = new_id[edge.a], new_id[edge.b]
        if rng.random() < 0.5:
            a, b = b, a
        edges.append(EdgeView(a, b, edge.travel_time))
    fleets = [
        replace(fleet, source_id=new_id[fleet.source_id], target_id=new_id[fleet.target_id])
        for fleet in state.fleets
    ]
    rng.shuffle(bases)
    rng.shuffle(edges)
    rng.shuffle(fleets)
    return replace(state, bases=tuple(bases), edges=tuple(edges), fleets=tuple(fleets)), new_id


def colours_swapped(state):
    """The same position with Red and Blue exchanged, seen by the same seat."""

    return replace(
        state,
        player_id=-state.player_id,
        bases=tuple(replace(base, owner=-base.owner) for base in state.bases),
        fleets=tuple(replace(fleet, owner=-fleet.owner) for fleet in state.fleets),
    )


class StateEncoderTests(unittest.TestCase):
    def test_shapes(self):
        state = played_game(generate_map(3).definition).view_for(RED)
        encoded = encode_state(state)
        count = len(state.bases)
        self.assertEqual(encoded.base_features.shape, (count, len(BASE_FEATURES)))
        self.assertEqual(encoded.pair_features.shape, (count, count, len(PAIR_FEATURES)))
        self.assertEqual(encoded.route_features.shape, (count, count, len(ROUTE_FEATURES)))
        self.assertEqual(encoded.global_features.shape, (len(GLOBAL_FEATURES),))
        self.assertEqual(encoded.base_ids, tuple(base.id for base in state.bases))

    def test_values_are_finite_and_modest(self):
        for definition in two_player_definitions():
            game = played_game(definition, ticks=600)
            for player in (RED, BLUE):
                encoded = encode_state(game.view_for(player))
                tables = (encoded.base_features, encoded.pair_features, encoded.route_features, encoded.global_features)
                # "forward" is the one feature that can be negative: a route leading away from the front.
                forward = ROUTE_FEATURES.index("forward")
                route_features = np.delete(encoded.route_features, forward, axis=-1)
                tables = (encoded.base_features, encoded.pair_features, route_features, encoded.global_features)
                for table in tables:
                    self.assertTrue(np.isfinite(table).all(), definition.name)
                    self.assertGreaterEqual(table.min(), 0.0)
                    self.assertLess(table.max(), 10.0)
                self.assertLess(np.abs(encoded.route_features[..., forward]).max(), 10.0)

    def test_each_base_has_exactly_one_owner_now_and_in_future(self):
        encoded = encode_state(played_game(generate_map(5).definition).view_for(BLUE))
        now = encoded.base_features[:, column("is_mine") : column("is_neutral") + 1]
        future = encoded.base_features[:, column("future_is_mine") : column("future_is_neutral") + 1]
        self.assertTrue((now.sum(axis=1) == 1).all())
        self.assertTrue((future.sum(axis=1) == 1).all())
        for ticks in HORIZON_TICKS:
            later = encoded.base_features[:, column(f"in_{ticks}_ticks_is_mine") : column(f"in_{ticks}_ticks_is_neutral") + 1]
            self.assertTrue((later.sum(axis=1) == 1).all(), ticks)

    def test_opponents_see_each_other_as_the_enemy(self):
        game = played_game(generate_map(5).definition)
        red = encode_state(game.view_for(RED)).base_features
        blue = encode_state(game.view_for(BLUE)).base_features
        np.testing.assert_array_equal(red[:, column("is_mine")], blue[:, column("is_enemy")])
        np.testing.assert_array_equal(red[:, column("incoming_mine")], blue[:, column("incoming_enemy")])
        np.testing.assert_array_equal(red[:, column("future_is_mine")], blue[:, column("future_is_enemy")])
        np.testing.assert_array_equal(red[:, column("units")], blue[:, column("units")])

    def test_feature_sets_name_existing_columns(self):
        tables = {
            "base_features": BASE_FEATURES,
            "pair_features": PAIR_FEATURES,
            "route_features": ROUTE_FEATURES,
            "global_features": GLOBAL_FEATURES,
        }
        for name, chosen in FEATURE_SETS.items():
            self.assertEqual(set(chosen), set(tables), name)
            for table, columns in chosen.items():
                self.assertTrue(set(columns) <= set(tables[table]), (name, table))
                self.assertEqual(len(set(columns)), len(columns), (name, table))
        # v1 is what our networks up to 2026-10-08 read: the first 14, 3, 0 and 4 columns.
        self.assertEqual([len(FEATURE_SETS["v1"][table]) for table in tables], [14, 3, 0, 4])
        self.assertEqual([len(FEATURE_SETS["v2"][table]) for table in tables], [34, 3, 2, 4])

    def test_plain_counts_match_the_logarithmic_ones(self):
        state = played_game(generate_map(3).definition, ticks=300).view_for(RED)
        encoded = encode_state(state).base_features
        pairs = (
            ("units", "units_linear"),
            ("incoming_mine", "incoming_mine_linear"),
            ("incoming_enemy", "incoming_enemy_linear"),
            ("future_units", "future_units_linear"),
        )
        for logarithmic, plain in pairs:
            np.testing.assert_allclose(
                encoded[:, column(plain)] * 100, np.expm1(encoded[:, column(logarithmic)] / _UNITS_SCALE), rtol=1e-4, atol=1e-3
            )
        production = {base.id: base.production_rate for base in state.bases}
        np.testing.assert_allclose(
            encoded[:, column("production_linear")] * 10, [production[base_id] for base_id in encode_state(state).base_ids], rtol=1e-6
        )

    def test_forecasts_match_the_game_played_on_without_new_sends(self):
        checked = 0
        for definition in two_player_definitions()[:6]:
            game = played_game(definition)
            for player in (RED, BLUE):
                encoded = encode_state(game.view_for(player))
                rows = {base_id: row for row, base_id in enumerate(encoded.base_ids)}
                later = copy.deepcopy(game)
                for ticks in range(1, max(HORIZON_TICKS) + 1):
                    later.step({})
                    if ticks not in HORIZON_TICKS:
                        continue
                    for base in later.view_for(player).bases:
                        features = encoded.base_features[rows[base.id]]
                        owner = "is_mine" if base.owner == player else "is_neutral" if base.owner == NEUTRAL else "is_enemy"
                        self.assertEqual(features[column(f"in_{ticks}_ticks_{owner}")], 1.0, (definition.name, ticks, base.id))
                        self.assertAlmostEqual(features[column(f"in_{ticks}_ticks_units")] * 100, base.units, delta=1e-3)
                        checked += 1
        self.assertGreater(checked, 500)

    def test_front_bases_and_distance_to_the_enemy(self):
        for definition in two_player_definitions():
            state = played_game(definition, ticks=300).view_for(RED)
            encoded = encode_state(state)
            seconds = shortest_seconds(state)
            owner = {base.id: base.owner for base in state.bases}
            enemy = [base_id for base_id, who in owner.items() if who not in (RED, NEUTRAL)]
            neighbours = {base_id: set() for base_id in owner}
            for edge in state.edges:
                neighbours[edge.a].add(edge.b)
                neighbours[edge.b].add(edge.a)
            for base_id, features in zip(encoded.base_ids, encoded.base_features):
                front = any(owner[other] != RED for other in neighbours[base_id])
                self.assertEqual(features[column("has_non_mine_neighbour")], float(front), (definition.name, base_id))
                if not enemy:  # the enemy has lost its last base (see the next test)
                    continue
                nearest = min(seconds[base_id][other] for other in enemy)
                self.assertAlmostEqual(features[column("distance_to_enemy")] * PATH_SECONDS_REFERENCE, nearest, places=4)

    def test_without_an_enemy_base_everything_is_as_far_as_the_map_allows(self):
        state = played_game(generate_map(3).definition).view_for(RED)
        lonely = replace(state, bases=tuple(replace(base, owner=NEUTRAL) if base.owner == BLUE else base for base in state.bases))
        distance = encode_state(lonely).base_features[:, column("distance_to_enemy")]
        seconds = shortest_seconds(lonely)
        longest = max(max(row.values()) for row in seconds.values())
        np.testing.assert_allclose(distance * PATH_SECONDS_REFERENCE, longest, rtol=1e-5)

    def test_forward_says_how_much_nearer_the_enemy_a_route_leads(self):
        forward = ROUTE_FEATURES.index("forward")
        checked = 0
        for definition in two_player_definitions():
            state = played_game(definition, ticks=300).view_for(BLUE)
            encoded = encode_state(state)
            seconds = shortest_seconds(state)
            owner = {base.id: base.owner for base in state.bases}
            row = {base_id: index for index, base_id in enumerate(encoded.base_ids)}
            for edge in state.edges:
                for source, target in ((edge.a, edge.b), (edge.b, edge.a)):
                    value = encoded.route_features[row[source], row[target], forward]
                    if owner[source] == NEUTRAL:
                        self.assertEqual(value, 0.0)
                        continue
                    # From the source owner's point of view: towards its own enemies.
                    enemy = [b for b, who in owner.items() if who not in (owner[source], NEUTRAL)]
                    if not enemy:  # no enemy base left: no direction is forward
                        self.assertEqual(value, 0.0)
                        continue
                    nearer = min(seconds[source][e] for e in enemy) - min(seconds[target][e] for e in enemy)
                    self.assertAlmostEqual(value * PATH_SECONDS_REFERENCE, nearer, places=4)
                    checked += 1
        self.assertGreater(checked, 50)

    def test_names_order_and_coordinates_do_not_matter(self):
        for definition in two_player_definitions():
            state = played_game(definition).view_for(RED)
            original = encode_state(state)
            changed_state, new_id = renamed_and_shuffled(state, seed=1)
            changed = encode_state(changed_state)

            # Line the changed rows up with the original rows, then everything must match.
            rows = [changed.base_ids.index(new_id[base_id]) for base_id in original.base_ids]
            np.testing.assert_array_equal(changed.base_features[rows], original.base_features)
            np.testing.assert_array_equal(changed.pair_features[rows][:, rows], original.pair_features)
            np.testing.assert_array_equal(changed.route_features[rows][:, rows], original.route_features)
            np.testing.assert_array_equal(changed.global_features, original.global_features)

    def test_swapping_colours_changes_nothing(self):
        for definition in two_player_definitions():
            game = played_game(definition)
            for player in (RED, BLUE):
                state = game.view_for(player)
                original = encode_state(state)
                swapped = encode_state(colours_swapped(state))
                np.testing.assert_array_equal(swapped.base_features, original.base_features)
                np.testing.assert_array_equal(swapped.route_features, original.route_features)
                np.testing.assert_array_equal(swapped.global_features, original.global_features)

    def test_both_seats_see_the_same_start_on_a_mirrored_map(self):
        for seed in range(10):
            definition = generate_map(seed, asymmetry=0.0).definition
            game = Game(definition)
            red = encode_state(game.view_for(RED))
            blue = encode_state(game.view_for(BLUE))

            # Each base's mirror image sits at the same height on the other side.
            position = {(base.x, base.y): base.id for base in definition.bases}
            mirror = {base.id: position[(definition.width - base.x, base.y)] for base in definition.bases}
            rows = [blue.base_ids.index(mirror[base_id]) for base_id in red.base_ids]

            np.testing.assert_array_equal(blue.base_features[rows], red.base_features)
            np.testing.assert_array_equal(blue.pair_features[rows][:, rows], red.pair_features)
            np.testing.assert_array_equal(blue.route_features[rows][:, rows], red.route_features)
            np.testing.assert_array_equal(blue.global_features, red.global_features)

    def test_pair_features_describe_the_routes(self):
        definition = load_map("classic")
        encoded = encode_state(Game(definition).view_for(RED))
        row = {base_id: index for index, base_id in enumerate(encoded.base_ids)}
        west, north, centre, east = row["west_home"], row["north"], row["center"], row["east_home"]
        connected, travel, shortest = (PAIR_FEATURES.index(name) for name in PAIR_FEATURES)

        self.assertEqual(encoded.pair_features[west, north, connected], 1.0)
        self.assertEqual(encoded.pair_features[north, west, connected], 1.0)
        self.assertEqual(encoded.pair_features[west, east, connected], 0.0)
        self.assertAlmostEqual(encoded.pair_features[north, centre, travel], 3 / TRAVEL_SECONDS_REFERENCE)
        # No direct route between the two homes; the fastest way is 5s + 5s through any middle base.
        self.assertEqual(encoded.pair_features[west, east, travel], 0.0)
        self.assertAlmostEqual(
            encoded.pair_features[west, east, shortest],
            encoded.pair_features[west, north, shortest] * 2,
        )
        self.assertEqual(encoded.pair_features[west, west, shortest], 0.0)

    def test_fleets_show_up_at_their_target(self):
        game = played_game(generate_map(3).definition)
        state = game.view_for(RED)
        self.assertTrue(state.fleets)
        encoded = encode_state(state)
        targets = {fleet.target_id for fleet in state.fleets}
        for base_id, row in zip(encoded.base_ids, encoded.base_features):
            has_incoming = row[column("incoming_mine")] + row[column("incoming_enemy")] > 0
            self.assertEqual(has_incoming, base_id in targets)
            arrives_soon = row[column("mine_arrives_soon")] + row[column("enemy_arrives_soon")] > 0
            self.assertEqual(arrives_soon, base_id in targets)

    def test_route_features_only_on_routes(self):
        for definition in two_player_definitions():
            encoded = encode_state(played_game(definition).view_for(RED))
            connected = encoded.pair_features[..., PAIR_FEATURES.index("connected")] == 1
            self.assertFalse(encoded.route_features[~connected].any(), definition.name)
            # Somewhere on the map, a capture costs something.
            self.assertTrue(encoded.route_features[connected].any(), definition.name)

    def test_capture_cost_agrees_with_just_enough(self):
        cost = ROUTE_FEATURES.index("capture_cost")
        share = ROUTE_FEATURES.index("capture_cost_share")
        plain = ROUTE_FEATURES.index("capture_cost_linear")
        checked = 0
        for definition in two_player_definitions():
            game = played_game(definition, ticks=300)
            for player in (RED, BLUE):
                state = game.view_for(player)
                encoded = encode_state(state)
                legal = legal_actions(state)
                units = {row: int(base.units) for row, base in enumerate(state.bases)}
                for source, target in zip(*np.nonzero(legal.allowed[..., JUST_ENOUGH])):
                    needed = legal.units[source, target, JUST_ENOUGH]
                    values = encoded.route_features[source, target]
                    self.assertAlmostEqual(np.expm1(values[cost] / _UNITS_SCALE), needed, places=3)
                    self.assertAlmostEqual(values[share], needed / units[source], places=5)
                    self.assertAlmostEqual(values[plain] * 100, needed, places=3)
                    checked += 1
        self.assertGreater(checked, 20)

    def test_routes_out_of_neutral_bases_are_empty(self):
        state = Game(generate_map(3).definition).view_for(RED)
        encoded = encode_state(state)
        neutral = [row for row, base in enumerate(state.bases) if base.owner == 0]
        self.assertTrue(neutral)
        self.assertFalse(encoded.route_features[neutral].any())

    def test_production_shares_of_all(self):
        game = played_game(generate_map(3).definition, ticks=300)
        for player in (RED, BLUE):
            shares = dict(zip(GLOBAL_FEATURES, encode_state(game.view_for(player)).global_features))
            total = (
                shares["my_production_of_all"]
                + shares["strongest_enemy_production_of_all"]
                + shares["second_enemy_production_of_all"]
                + shares["neutral_production_share"]
            )
            self.assertAlmostEqual(total, 1.0, places=5)
            self.assertEqual(shares["second_enemy_production_of_all"], 0.0)
            # The older two-player share is the same split without the neutral bases.
            players = shares["my_production_of_all"] + shares["strongest_enemy_production_of_all"]
            self.assertAlmostEqual(shares["my_production_share"], shares["my_production_of_all"] / players, places=5)

    def test_global_shares(self):
        game = Game(generate_map(3, asymmetry=0.0).definition)
        start = dict(zip(GLOBAL_FEATURES, encode_state(game.view_for(RED)).global_features))
        self.assertAlmostEqual(start["my_units_share"], 0.5)
        self.assertAlmostEqual(start["my_production_share"], 0.5)
        self.assertEqual(start["time"], 0.0)

        game = played_game(generate_map(3).definition, ticks=300)
        red = dict(zip(GLOBAL_FEATURES, encode_state(game.view_for(RED)).global_features))
        blue = dict(zip(GLOBAL_FEATURES, encode_state(game.view_for(BLUE)).global_features))
        self.assertAlmostEqual(red["my_units_share"] + blue["my_units_share"], 1.0, places=6)
        self.assertAlmostEqual(red["my_production_share"] + blue["my_production_share"], 1.0, places=6)
        self.assertAlmostEqual(red["time"], blue["time"])

    def test_explicit_layout_gives_the_same_result(self):
        game = played_game(generate_map(7).definition)
        state = game.view_for(RED)
        layout = build_layout(state)
        with_layout = encode_state(state, layout)
        without = encode_state(state)
        np.testing.assert_array_equal(with_layout.base_features, without.base_features)
        np.testing.assert_array_equal(with_layout.pair_features, without.pair_features)

    def test_stacking_pads_smaller_maps(self):
        states = [
            encode_state(Game(generate_map(seed).definition).view_for(RED))
            for seed in range(6)
        ]
        sizes = [len(state.base_ids) for state in states]
        self.assertGreater(len(set(sizes)), 1)

        batch = stack_states(states)
        largest = max(sizes)
        self.assertEqual(batch.base_features.shape, (6, largest, len(BASE_FEATURES)))
        self.assertEqual(batch.pair_features.shape, (6, largest, largest, len(PAIR_FEATURES)))
        self.assertEqual(batch.global_features.shape, (6, len(GLOBAL_FEATURES)))
        for row, (state, size) in enumerate(zip(states, sizes)):
            self.assertEqual(batch.base_mask[row].sum(), size)
            np.testing.assert_array_equal(batch.base_features[row, :size], state.base_features)
            np.testing.assert_array_equal(batch.pair_features[row, :size, :size], state.pair_features)
            self.assertEqual(np.abs(batch.base_features[row, size:]).sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
