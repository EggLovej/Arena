import unittest
from dataclasses import FrozenInstanceError

from game import Game
from game_types import (
    BLUE,
    GREEN,
    NEUTRAL,
    RED,
    BaseDefinition,
    EdgeDefinition,
    MapDefinition,
    SendAction,
)
from loading import available_ais, available_maps, load_ai, load_map, validate_map
from main import collect_actions, play_headless
from human_controller import HumanController
from sizing import base_radii


def small_map(target_units=5):
    return MapDefinition(
        name="test",
        width=300,
        height=100,
        bases=(
            BaseDefinition("left", RED, 10, 0, 20, 50),
            BaseDefinition("middle", NEUTRAL, target_units, 0, 150, 50),
            BaseDefinition("right", BLUE, 10, 0, 280, 50),
        ),
        edges=(
            EdgeDefinition("left", "middle", 1),
            EdgeDefinition("right", "middle", 1),
        ),
    )


def stream_map():
    """Two red bases that produce, both connected to a neutral one."""

    return MapDefinition(
        name="streams",
        width=300,
        height=100,
        bases=(
            BaseDefinition("left", RED, 10, 2, 20, 50),
            BaseDefinition("top", RED, 10, 2, 150, 10),
            BaseDefinition("middle", NEUTRAL, 5, 0, 150, 50),
            BaseDefinition("right", BLUE, 10, 0, 280, 50),
        ),
        edges=(
            EdgeDefinition("left", "middle", 1),
            EdgeDefinition("top", "middle", 1),
            EdgeDefinition("right", "middle", 1),
        ),
    )


class GameTests(unittest.TestCase):
    def test_simultaneous_tie_neutralizes_target(self):
        game = Game(small_map(), seconds_per_tick=1)
        game.step(
            {
                RED: SendAction("left", "middle", 7),
                BLUE: SendAction("right", "middle", 7),
            }
        )
        self.assertEqual(game.bases["middle"].owner, NEUTRAL)
        self.assertEqual(game.bases["middle"].units, 0)

    def test_action_dictionary_order_does_not_change_result(self):
        first = Game(small_map(), seconds_per_tick=1)
        second = Game(small_map(), seconds_per_tick=1)

        first.step(
            {
                RED: SendAction("left", "middle", 7),
                BLUE: SendAction("right", "middle", 6),
            }
        )
        second.step(
            {
                BLUE: SendAction("right", "middle", 6),
                RED: SendAction("left", "middle", 7),
            }
        )
        self.assertEqual(first.view_for(RED), second.view_for(RED))

    def test_zero_unit_action_is_rejected_without_creating_a_fleet(self):
        game = Game(small_map())
        errors = game.step({RED: SendAction("left", "middle", 0), BLUE: None})
        self.assertIn(RED, errors)
        self.assertEqual(game.fleets, [])

    def test_ai_receives_an_immutable_snapshot(self):
        game = Game(small_map())
        state = game.view_for(RED)
        with self.assertRaises(FrozenInstanceError):
            state.bases[0].units = 999

    def test_maps_are_selected_from_the_maps_directory(self):
        self.assertEqual(
            available_maps(),
            (
                "classic",
                "clockwork_citadel",
                "crossroads",
                "four_corners",
                "frontier",
                "nomads",
                "ring",
                "salt_and_pepper",
                "shattered_spiral",
                "siege",
                "tectonic_archipelago",
                "three_fronts",
                "threefold_crucible",
                "triangle",
                "triple_ring",
            ),
        )
        self.assertEqual(load_map("ring").name, "ring")

    def test_new_maps_have_requested_sizes(self):
        self.assertEqual(len(load_map("crossroads").bases), 10)
        self.assertEqual(len(load_map("frontier").bases), 19)
        self.assertEqual(len(load_map("four_corners").bases), 30)

    def test_nomads_start_with_armies_but_no_production(self):
        owned = [base for base in load_map("nomads").bases if base.owner != NEUTRAL]
        self.assertEqual({base.owner for base in owned}, {RED, BLUE})
        for base in owned:
            self.assertEqual((base.units, base.production_rate), (80, 0))

    def test_salt_and_pepper_starts_with_five_bases_each(self):
        owners = [base.owner for base in load_map("salt_and_pepper").bases]
        self.assertEqual(owners.count(RED), 5)
        self.assertEqual(owners.count(BLUE), 5)

    def test_siege_pits_one_fortress_against_four_camps(self):
        owners = [base.owner for base in load_map("siege").bases]
        self.assertEqual(owners.count(RED), 1)
        self.assertEqual(owners.count(BLUE), 4)

    def test_symmetric_maps_give_both_players_identical_positions(self):
        # Nomads and Tectonic Archipelago are mirrored left-right; Salt & Pepper and
        # Clockwork Citadel are rotated 180 degrees.
        for map_name, rotate in (
            ("nomads", False),
            ("salt_and_pepper", True),
            ("tectonic_archipelago", False),
            ("clockwork_citadel", True),
        ):
            with self.subTest(map_name):
                map_definition = load_map(map_name)
                xs = [base.x for base in map_definition.bases]
                ys = [base.y for base in map_definition.bases]
                by_position = {(base.x, base.y): base for base in map_definition.bases}
                swapped = {RED: BLUE, BLUE: RED, GREEN: GREEN, NEUTRAL: NEUTRAL}

                twin_ids = {}
                for base in map_definition.bases:
                    x = min(xs) + max(xs) - base.x
                    y = min(ys) + max(ys) - base.y if rotate else base.y
                    twin = by_position[(x, y)]
                    self.assertEqual(
                        (twin.owner, twin.units, twin.production_rate),
                        (swapped[base.owner], base.units, base.production_rate),
                    )
                    twin_ids[base.id] = twin.id

                travel_times = {
                    frozenset((edge.a, edge.b)): edge.travel_time
                    for edge in map_definition.edges
                }
                for edge in map_definition.edges:
                    twin_edge = frozenset((twin_ids[edge.a], twin_ids[edge.b]))
                    self.assertEqual(travel_times.get(twin_edge), edge.travel_time)

    def test_large_maps_play_for_ten_minutes(self):
        for map_name in ("clockwork_citadel", "shattered_spiral", "tectonic_archipelago", "threefold_crucible"):
            with self.subTest(map_name):
                map_definition = load_map(map_name)
                self.assertGreaterEqual(len(map_definition.bases), 100)
                self.assertEqual(map_definition.time_limit, 600)
                self.assertTrue(map_definition.tagline)

    def test_threefold_crucible_guards_its_prize_behind_fortified_bridges(self):
        bases = load_map("threefold_crucible").bases
        prize = max(bases, key=lambda base: base.production_rate)
        self.assertEqual((prize.id, prize.production_rate, prize.units), ("crucible", 10.0, 1000))
        bridges = [base for base in bases if base.id.startswith("bridge")]
        self.assertEqual(len(bridges), 9)
        for bridge in bridges:
            self.assertEqual((bridge.production_rate, bridge.units), (0.2, 80))

    def test_four_corners_starts_in_opposite_corners(self):
        map_definition = load_map("four_corners")

        def closest_to(x, y):
            return min(
                map_definition.bases,
                key=lambda base: (base.x - x) ** 2 + (base.y - y) ** 2,
            )

        self.assertEqual(closest_to(0, 0).owner, BLUE)
        self.assertEqual(closest_to(map_definition.width, 0).owner, RED)
        self.assertEqual(closest_to(0, map_definition.height).owner, RED)
        self.assertEqual(
            closest_to(map_definition.width, map_definition.height).owner,
            BLUE,
        )

    def test_three_player_maps_activate_red_blue_and_green(self):
        for map_name in ("triangle", "three_fronts", "triple_ring"):
            game = Game(load_map(map_name))
            self.assertEqual(game.players, (RED, BLUE, GREEN))

    def test_collect_actions_calls_all_three_controllers(self):
        game = Game(load_map("triangle"))
        called_players = []

        def controller(state):
            called_players.append(state.player_id)
            return None

        actions = collect_actions(game, controller, controller, controller)

        self.assertEqual(set(called_players), {RED, BLUE, GREEN})
        self.assertEqual(set(actions), {RED, BLUE, GREEN})

    def test_ais_are_discovered_from_the_ais_directory(self):
        self.assertEqual(
            available_ais(),
            ("aggressive_ai", "atlas_ai", "atlas_clean_ai", "beast_ai", "careful_ai", "neural_ai", "strategic_ai"),
        )

    def test_strategic_ai_routes_units_out_of_an_enclosed_base(self):
        map_definition = MapDefinition(
            name="enclosed-base-test",
            width=400,
            height=100,
            bases=(
                BaseDefinition("reserve", RED, 20, 1, 30, 50),
                BaseDefinition("front", RED, 3, 1, 140, 50),
                BaseDefinition("neutral", NEUTRAL, 8, 1, 260, 50),
                BaseDefinition("enemy", BLUE, 20, 1, 370, 50),
            ),
            edges=(
                EdgeDefinition("reserve", "front", 2),
                EdgeDefinition("front", "neutral", 2),
                EdgeDefinition("neutral", "enemy", 2),
            ),
        )
        game = Game(map_definition)
        strategic_ai = load_ai("strategic_ai")

        action = strategic_ai(game.view_for(RED))

        self.assertEqual(action.source_id, "reserve")
        self.assertEqual(action.target_id, "front")
        self.assertGreater(action.units, 0)

    def test_invalid_travel_time_is_rejected(self):
        definition = small_map()
        invalid = MapDefinition(
            name=definition.name,
            width=definition.width,
            height=definition.height,
            bases=definition.bases,
            edges=(EdgeDefinition("left", "middle", 0),),
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_map(invalid)

    def test_sample_ais_run_on_every_map_and_replay_deterministically(self):
        red_ai = load_ai("aggressive_ai")
        blue_ai = load_ai("careful_ai")
        green_ai = load_ai("strategic_ai")

        for map_name in available_maps():
            first = Game(load_map(map_name))
            second = Game(load_map(map_name))
            play_headless(first, red_ai, blue_ai, 500, green_ai)
            play_headless(second, red_ai, blue_ai, 500, green_ai)
            self.assertEqual(first.view_for(RED), second.view_for(RED))

    def test_human_clicks_create_one_half_unit_action(self):
        game = Game(small_map())
        human = HumanController(RED)

        human.handle_click((20, 50), 1, game)
        self.assertEqual(human.selected_base_id, "left")
        human.handle_click((150, 50), 1, game)

        self.assertEqual(
            human.choose_action(game.view_for(RED)),
            SendAction("left", "middle", 5),
        )
        self.assertIsNone(human.choose_action(game.view_for(RED)))

    def test_human_shift_click_starts_a_stream_of_all_units(self):
        game = Game(stream_map())
        human = HumanController(RED)

        human.handle_click((20, 50), 1, game)
        human.handle_click((150, 50), 1, game, shift_held=True)
        self.assertEqual(human.streams, {"left": "middle"})

        self.assertEqual(human.choose_action(game.view_for(RED)), SendAction("left", "middle", 10))
        game.step({RED: SendAction("left", "middle", 10)})
        # The next unit produced goes the same way, without another click.
        while int(game.bases["left"].units) < 1:
            self.assertIsNone(human.choose_action(game.view_for(RED)))
            game.step({})
        self.assertEqual(human.choose_action(game.view_for(RED)), SendAction("left", "middle", 1))

    def test_human_shift_click_on_the_source_stops_its_stream(self):
        game = Game(stream_map())
        human = HumanController(RED)
        human.handle_click((20, 50), 1, game)
        human.handle_click((150, 50), 1, game, shift_held=True)

        human.handle_click((20, 50), 1, game, shift_held=True)

        self.assertEqual(human.streams, {})
        self.assertIsNone(human.selected_base_id)
        self.assertIsNone(human.choose_action(game.view_for(RED)))

    def test_a_click_goes_before_a_stream_and_streams_take_turns(self):
        game = Game(stream_map())
        human = HumanController(RED)
        human.streams = {"left": "middle", "top": "middle"}

        human.handle_click((150, 10), 1, game)  # top
        human.handle_click((150, 50), 1, game)  # middle: half of top's 10 units
        self.assertEqual(human.choose_action(game.view_for(RED)), SendAction("top", "middle", 5))
        first = human.choose_action(game.view_for(RED))
        second = human.choose_action(game.view_for(RED))
        self.assertEqual({first.source_id, second.source_id}, {"left", "top"})

    def test_a_stream_ends_when_its_source_is_lost(self):
        game = Game(stream_map())
        human = HumanController(RED)
        human.streams = {"left": "middle"}
        game.bases["left"].owner = BLUE
        self.assertIsNone(human.choose_action(game.view_for(RED)))
        self.assertEqual(human.streams, {})

    def test_human_cannot_select_an_opponents_base(self):
        game = Game(small_map())
        human = HumanController(RED)
        human.handle_click((280, 50), 1, game)
        self.assertIsNone(human.selected_base_id)

    def test_richer_bases_are_drawn_larger(self):
        game = Game(load_map("classic"))
        radii = base_radii(game.bases.values())

        # south (0.2/s) < north (1.0/s) < west_home (1.15/s) < center (1.45/s)
        self.assertLess(radii["south"], radii["north"])
        self.assertLess(radii["north"], radii["west_home"])
        self.assertLess(radii["west_home"], radii["center"])

    def test_human_click_near_the_edge_of_a_large_base_selects_it(self):
        game = Game(load_map("classic"))
        human = HumanController(RED)
        home = game.bases["west_home"]
        radius = base_radii(game.bases.values())[home.id]

        human.handle_click((home.x + radius - 2, home.y), 1, game)

        self.assertEqual(human.selected_base_id, home.id)

    def test_human_can_control_green_on_a_three_player_map(self):
        game = Game(load_map("triangle"))
        human = HumanController(GREEN)
        green_base = next(
            base for base in game.bases.values() if base.owner == GREEN
        )

        human.handle_click((green_base.x, green_base.y), 1, game)

        self.assertEqual(human.selected_base_id, green_base.id)


if __name__ == "__main__":
    unittest.main()
