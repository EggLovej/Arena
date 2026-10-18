import copy
import unittest

from game import Game
from game_types import BLUE, NEUTRAL, RED, BaseDefinition, EdgeDefinition, MapDefinition, SendAction
from loading import load_ai

from ml.encoding.forecast import BaseTimeline, ticks_until_arrival
from ml.map_generator import generate_map


def three_base_map():
    return MapDefinition(
        name="test",
        width=300,
        height=100,
        bases=(
            BaseDefinition("left", RED, 20, 0, 20, 50),
            BaseDefinition("middle", NEUTRAL, 5, 1.0, 150, 50),
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

    def test_equal_forces_leave_the_base_neutral_and_empty(self):
        game = Game(three_base_map())
        game.step({RED: SendAction("left", "middle", 5)})
        settled = forecast_from_game(game, "middle")
        self.assertEqual((settled.owner, settled.units), (NEUTRAL, 0.0))

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
