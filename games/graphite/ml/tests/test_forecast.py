import copy
import unittest

from game import Game
from game_types import BLUE, NEUTRAL, RED, BaseDefinition, EdgeDefinition, MapDefinition, SendAction
from loading import load_ai

from ml.encoding.forecast import BaseTimeline, ticks_until_arrival
from ml.map_generator import generate_map


def three_base_map(middle_owner=NEUTRAL, middle_units=5, production=1.0):
    return MapDefinition(
        name="test",
        width=300,
        height=100,
        bases=(
            BaseDefinition("left", RED, 20, 0, 20, 50),
            BaseDefinition("middle", middle_owner, middle_units, production, 150, 50),
            BaseDefinition("right", BLUE, 20, 0, 280, 50),
        ),
        edges=(
            EdgeDefinition("left", "middle", 0.5),
            EdgeDefinition("right", "middle", 0.8),
        ),
    )


def forecast_from_game(game, base_id, ticks=None):
    state = game.view_for(RED)
    incoming = [fleet for fleet in state.fleets if fleet.target_id == base_id]
    return BaseTimeline(state.get_base(base_id), incoming, state.seconds_per_tick).at(ticks)


def advance(game, ticks):
    """The real game, left alone for this many ticks."""

    future = copy.deepcopy(game)
    for _ in range(ticks):
        future.step({})
    return future


class ForecastTests(unittest.TestCase):
    def test_arrival_tick_matches_the_game(self):
        # A 0.1s route is left out: that fleet lands in the tick it departs and is never seen in flight.
        for travel_time in (0.2, 0.3, 0.5, 1.0, 2.3, 3.1, 7.9):
            game = Game(
                MapDefinition(
                    name="test",
                    width=300,
                    height=100,
                    bases=(
                        BaseDefinition("left", RED, 20, 0, 20, 50),
                        BaseDefinition("right", BLUE, 5, 0, 280, 50),
                    ),
                    edges=(EdgeDefinition("left", "right", travel_time),),
                )
            )
            game.step({RED: SendAction("left", "right", 10)})
            predicted = ticks_until_arrival(game.fleets[0].remaining_time, game.seconds_per_tick)
            actual = 0
            while game.fleets:
                game.step({})
                actual += 1
            self.assertEqual(predicted, actual, travel_time)

    def test_no_fleets_means_production_only(self):
        game = Game(three_base_map(middle_owner=BLUE, middle_units=5, production=2.0))
        forecast = forecast_from_game(game, "middle", ticks=10)
        self.assertEqual(forecast.owner, BLUE)
        self.assertAlmostEqual(forecast.units, 7.0)

        settled = forecast_from_game(game, "middle")
        self.assertEqual((settled.owner, settled.units, settled.ticks), (BLUE, 5, 0))

    def test_neutral_base_does_not_grow(self):
        game = Game(three_base_map())
        forecast = forecast_from_game(game, "middle", ticks=50)
        self.assertEqual((forecast.owner, forecast.units), (NEUTRAL, 5))

    def test_capture_then_production(self):
        game = Game(three_base_map())
        game.step({RED: SendAction("left", "middle", 8)})
        settled = forecast_from_game(game, "middle")
        self.assertEqual(settled.owner, RED)
        self.assertAlmostEqual(settled.units, 3.0)

        later = forecast_from_game(game, "middle", ticks=settled.ticks + 10)
        self.assertAlmostEqual(later.units, 4.0)

    def test_equal_forces_leave_the_base_neutral_and_empty(self):
        game = Game(three_base_map())
        game.step({RED: SendAction("left", "middle", 5)})
        settled = forecast_from_game(game, "middle")
        self.assertEqual((settled.owner, settled.units), (NEUTRAL, 0.0))

    def test_two_attackers_in_sequence(self):
        game = Game(three_base_map())
        game.step({RED: SendAction("left", "middle", 8), BLUE: SendAction("right", "middle", 9)})
        # Red lands first and holds 3, grows for 0.3s, then Blue's 9 arrive.
        settled = forecast_from_game(game, "middle")
        self.assertEqual(settled.owner, BLUE)
        self.assertAlmostEqual(settled.units, 9 - 3.3)

    def test_matches_the_real_game_on_generated_maps(self):
        strategic = load_ai("strategic_ai")
        aggressive = load_ai("aggressive_ai")
        checked = 0

        for seed in range(12):
            game = Game(generate_map(seed).definition)
            for tick in range(400):
                if game.is_finished():
                    break
                game.step({RED: strategic(game.view_for(RED)), BLUE: aggressive(game.view_for(BLUE))})
                if tick % 40 != 39 or not game.fleets:
                    continue

                targets = {fleet.target_id for fleet in game.fleets}
                for base_id in targets:
                    settled = forecast_from_game(game, base_id)
                    for ticks in (settled.ticks, 3, 17, settled.ticks + 25):
                        forecast = forecast_from_game(game, base_id, ticks)
                        actual = advance(game, ticks).bases[base_id]
                        self.assertEqual(forecast.owner, actual.owner, (seed, tick, base_id, ticks))
                        self.assertAlmostEqual(forecast.units, actual.units, places=6)
                        checked += 1

        self.assertGreater(checked, 500)


if __name__ == "__main__":
    unittest.main()
