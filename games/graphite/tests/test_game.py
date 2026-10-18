import importlib
import unittest
from dataclasses import replace
from types import SimpleNamespace

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
from main import collect_actions
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


def first_move(name, state):
    """A newly loaded AI's move. One that draws its moves by chance draws from seed 0, as in a seeded match."""

    module = importlib.import_module(f"ais.{name}")
    if hasattr(module, "create_bot"):
        bot = module.create_bot()
        bot.reset(SimpleNamespace(seed=0))
        return bot.act(state)
    return load_ai(name)(state)


class GameTests(unittest.TestCase):
    def test_simultaneous_arrivals_pool_each_side_whatever_the_order_of_actions(self):
        # The largest force takes what the runner-up cannot match; a tie leaves the base neutral and empty.
        for blue_units, owner, units in ((7, NEUTRAL, 0), (6, RED, 1)):
            for order in ((RED, BLUE), (BLUE, RED)):
                game = Game(small_map(), seconds_per_tick=1)
                sends = {RED: SendAction("left", "middle", 7), BLUE: SendAction("right", "middle", blue_units)}
                game.step({player: sends[player] for player in order})
                self.assertEqual((game.bases["middle"].owner, game.bases["middle"].units), (owner, units))

    def test_every_invalid_move_is_refused_and_changes_nothing(self):
        for action in (
            SendAction("left", "middle", 0),
            SendAction("left", "middle", 11),  # more than the base has
            SendAction("left", "middle", 5.0),
            SendAction("left", "middle", True),
            SendAction("left", "right", 5),  # not connected
            SendAction("right", "middle", 5),  # Blue's base
            SendAction("left", "left", 5),
            SendAction("left", "nowhere", 5),
            SendAction("nowhere", "middle", 5),
            ("left", "middle", 5),
        ):
            with self.subTest(action):
                game = Game(small_map())
                self.assertIn(RED, game.step({RED: action}))
                self.assertEqual(game.fleets, [])
                self.assertEqual(game.bases["left"].units, 10)

    def test_invalid_maps_are_refused(self):
        valid = small_map()
        left, *others = valid.bases
        for problem, invalid in (
            ("invalid owner", replace(valid, bases=(replace(left, owner=7), *others))),
            ("outside the map", replace(valid, bases=(replace(left, x=-1), *others))),
            ("negative", replace(valid, bases=(replace(left, units=-1), *others))),
            ("unknown base", replace(valid, edges=(*valid.edges, EdgeDefinition("left", "nowhere", 1)))),
            ("Duplicate edge", replace(valid, edges=(*valid.edges, EdgeDefinition("middle", "left", 2)))),
            ("travel times must be positive", replace(valid, edges=(EdgeDefinition("left", "middle", 0), valid.edges[1]))),
            ("Red and Blue", replace(valid, bases=(replace(left, owner=NEUTRAL), *others))),
            ("connected", replace(valid, edges=valid.edges[:1])),
        ):
            with self.subTest(problem), self.assertRaisesRegex(ValueError, problem):
                validate_map(invalid)

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

    def test_every_map_plays_without_a_refused_move(self):
        red, blue, green = load_ai("aggressive_ai"), load_ai("careful_ai"), load_ai("strategic_ai")
        for map_name in available_maps():
            game = Game(load_map(map_name))
            for _ in range(150):
                self.assertEqual(game.step(collect_actions(game, red, blue, green)), {}, map_name)

    def test_every_ai_loads_and_plays_every_map_without_a_refused_move(self):
        self.assertNotIn("_atlas_sim", available_ais())  # helpers are not players
        strategic = load_ai("strategic_ai")
        for name in available_ais():
            ai, sends = load_ai(name), 0
            for map_name in available_maps():
                game = Game(load_map(map_name))
                for _ in range(5):
                    actions = {player: (ai if player == RED else strategic)(game.view_for(player)) for player in game.players}
                    sends += actions[RED] is not None
                    self.assertEqual(game.step(actions), {}, (name, map_name))
            self.assertGreater(sends, 0, name)

    def test_renaming_and_moving_the_bases_does_not_change_any_ai_s_move(self):
        # AIs get arbitrary base ids and coordinates, so they must reason from the graph.
        state = Game(load_map("crossroads")).view_for(RED)
        ids = {base.id: f"unseen-{137 * number + 19}" for number, base in enumerate(state.bases)}
        renamed = replace(
            state,
            bases=tuple(replace(base, id=ids[base.id], x=3, y=9) for base in state.bases),
            edges=tuple(replace(edge, a=ids[edge.a], b=ids[edge.b]) for edge in state.edges),
        )
        for name in available_ais():
            with self.subTest(name):
                action = first_move(name, state)
                expected = action and SendAction(ids[action.source_id], ids[action.target_id], action.units)
                self.assertEqual(first_move(name, renamed), expected)

    def test_a_click_sends_half_of_one_of_your_own_bases(self):
        game = Game(small_map())
        human = HumanController(RED)
        human.handle_click((280, 50), 1, game)  # Blue's base
        self.assertIsNone(human.selected_base_id)

        human.handle_click((20, 50), 1, game)
        self.assertEqual(human.selected_base_id, "left")
        human.handle_click((150, 50), 1, game)
        self.assertEqual(human.choose_action(game.view_for(RED)), SendAction("left", "middle", 5))
        self.assertIsNone(human.choose_action(game.view_for(RED)))

    def test_shift_click_starts_a_stream_and_on_its_source_stops_it(self):
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

        human.handle_click((20, 50), 1, game, shift_held=True)
        self.assertEqual(human.streams, {})
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


if __name__ == "__main__":
    unittest.main()
