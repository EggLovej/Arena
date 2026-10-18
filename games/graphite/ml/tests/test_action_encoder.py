import copy
import unittest

import numpy as np

from game import Game
from game_types import BLUE, NEUTRAL, RED, BaseDefinition, EdgeDefinition, MapDefinition, SendAction

from ml.encoding.action_encoder import (
    AMOUNT_OPTIONS,
    HOLD,
    JUST_ENOUGH,
    SURPLUS,
    SEND_FRACTIONS,
    SendChoice,
    decode_action,
    encode_action,
    legal_actions,
)
from ml.encoding.forecast import ticks_until_arrival
from ml.encoding.state_encoder import layout_for
from ml.map_generator import generate_map
from ml.tests.support import scripted_game, scripted_games, two_player_maps


ALL = AMOUNT_OPTIONS.index("100%")
# The options that are always open on a usable route: the eighths.
EIGHTHS = slice(JUST_ENOUGH + 1, JUST_ENOUGH + 1 + len(SEND_FRACTIONS))


def line_map(middle_owner=NEUTRAL, middle_units=5, production=0.0):
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


def _kept(game, action, base_id):
    """Send ``action`` for Red, then let every fleet now in flight (and it) land with nobody
    sending anything: is ``base_id`` Red's from the send's landing on, at every tick?"""

    future = copy.deepcopy(game)
    landing = max((f.remaining_time for f in future.fleets), default=0.0)
    travel = future._travel_times[frozenset((action.source_id, action.target_id))]
    sent_lands = ticks_until_arrival(travel, future.seconds_per_tick)
    last = max(sent_lands, ticks_until_arrival(landing, future.seconds_per_tick) if landing else 0)
    future.step({RED: action})
    # The source is watched from the start, the target from the moment the fleet lands.
    first_watched = 1 if base_id == action.source_id else sent_lands
    for tick in range(1, last + 1):
        if tick > 1:
            future.step({})
        if tick >= first_watched and future.bases[base_id].owner != RED:
            return False
    return True


def choices(legal):
    return [SendChoice(*(int(i) for i in entry)) for entry in np.argwhere(legal.allowed)]


def row(legal, base_id):
    return legal.base_ids.index(base_id)


class ActionEncoderTests(unittest.TestCase):
    def test_opening_choices_on_a_small_map(self):
        legal = legal_actions(Game(line_map()).view_for(RED))
        left, middle, right = (row(legal, name) for name in ("left", "middle", "right"))

        # Only one route leaves Red's base, with "just enough" and every eighth open on it.
        # Nothing is on its way anywhere, so "hold" would be "just enough" and "surplus" all:
        # neither is offered.
        self.assertEqual(int(legal.allowed.any(axis=-1).sum()), 1)
        self.assertTrue(legal.allowed[left, middle, : EIGHTHS.stop].all())
        # Just enough to beat 5 defenders, then eighths of Red's 20 units, rounded up.
        self.assertEqual(list(legal.units[left, middle]), [6, 3, 5, 8, 10, 13, 15, 18, 20, 0, 0])
        self.assertFalse(legal.allowed[middle].any())
        self.assertFalse(legal.allowed[right].any())

    def test_illegal_choice_is_refused(self):
        legal = legal_actions(Game(line_map()).view_for(RED))
        with self.assertRaises(ValueError):
            decode_action(SendChoice(row(legal, "right"), row(legal, "middle"), ALL), legal)

    def test_just_enough_counts_growth_and_fleets_in_flight_and_hides_when_unaffordable(self):
        legal = legal_actions(Game(line_map(middle_units=20)).view_for(RED))
        left, middle = row(legal, "left"), row(legal, "middle")
        self.assertFalse(legal.allowed[left, middle, JUST_ENOUGH])
        self.assertTrue(legal.allowed[left, middle, EIGHTHS].all())

        # Blue's base has 5 units and gains 2 per second; Red's fleet flies for 0.5s.
        legal = legal_actions(Game(line_map(BLUE, 5, production=2.0)).view_for(RED))
        self.assertEqual(legal.units[left, middle, JUST_ENOUGH], 7)

        game = Game(line_map())
        game.step({RED: SendAction("left", "middle", 3)})
        # 3 of the 5 defenders are already dealt with by the time a second fleet lands.
        self.assertEqual(legal_actions(game.view_for(RED)).units[left, middle, JUST_ENOUGH], 3)
        game.step({RED: SendAction("left", "middle", 3)})
        # The two fleets in flight now take the base by themselves.
        self.assertFalse(legal_actions(game.view_for(RED)).allowed[left, middle, JUST_ENOUGH])

    def test_every_allowed_choice_is_accepted_by_the_game(self):
        checked = 0
        for definition in two_player_maps():
            for game in scripted_games(definition, every=50):
                for player in (RED, BLUE):
                    legal = legal_actions(game.view_for(player))
                    for choice in choices(legal):
                        action = decode_action(choice, legal)
                        self.assertIsNone(game._validate_action(player, action), (definition.name, action))
                        checked += 1
        self.assertGreater(checked, 5000)

    def test_every_route_the_game_accepts_is_offered(self):
        for definition in two_player_maps():
            for game in scripted_games(definition, every=50):
                for player in (RED, BLUE):
                    legal = legal_actions(game.view_for(player))
                    offered = {
                        (legal.base_ids[source], legal.base_ids[target])
                        for source, target in np.argwhere(legal.allowed.any(axis=-1))
                    }
                    accepted = {
                        (a.id, b.id)
                        for a in game.bases.values()
                        for b in game.bases.values()
                        if game._validate_action(player, SendAction(a.id, b.id, 1)) is None
                    }
                    self.assertEqual(offered, accepted, definition.name)

    def test_just_enough_takes_the_base_and_one_fewer_does_not(self):
        captures = near_misses = 0
        for definition in two_player_maps():
            for game in scripted_games(definition, every=20):
                state = game.view_for(RED)
                layout = layout_for(state)
                legal = legal_actions(state, layout)
                for choice in choices(legal):
                    if choice.amount != JUST_ENOUGH:
                        continue
                    action = decode_action(choice, legal)
                    travel = dict(layout.neighbours[choice.source])[choice.target]
                    ticks = ticks_until_arrival(travel, state.seconds_per_tick)

                    # Send it, then leave the real game alone until the fleet lands.
                    future = copy.deepcopy(game)
                    future.step({RED: action})
                    for _ in range(ticks - 1):
                        future.step({})
                    self.assertEqual(future.bases[action.target_id].owner, RED, (definition.name, action))
                    captures += 1

                    if action.units > 1:
                        short = copy.deepcopy(game)
                        short.step({RED: SendAction(action.source_id, action.target_id, action.units - 1)})
                        for _ in range(ticks - 1):
                            short.step({})
                        near_misses += short.bases[action.target_id].owner != RED
        self.assertGreater(captures, 300)
        # One fewer fails except in the rare case covered by the safety margin.
        self.assertGreater(near_misses, 0.95 * captures)

    def test_hold_keeps_the_target_through_every_fleet_on_its_way_and_one_fewer_does_not(self):
        holds = tried_fewer = near_misses = 0
        for definition in two_player_maps():
            # "hold" differs from "just enough" only with fleets on their way: look often.
            for game in scripted_games(definition, every=5):
                state = game.view_for(RED)
                legal = legal_actions(state)
                for choice in choices(legal):
                    if choice.amount != HOLD:
                        continue
                    action = decode_action(choice, legal)
                    holds += 1
                    self.assertTrue(_kept(game, action, action.target_id), (definition.name, action))
                    if action.units > 1:
                        tried_fewer += 1
                        fewer = SendAction(action.source_id, action.target_id, action.units - 1)
                        near_misses += not _kept(game, fewer, action.target_id)
        self.assertGreater(holds, 30)
        # One fewer fails except in the rare case covered by the safety margin.
        self.assertGreater(near_misses, 0.95 * tried_fewer)

    def test_surplus_leaves_the_source_safe_and_one_more_does_not(self):
        offered = near_misses = 0
        for definition in two_player_maps():
            for game in scripted_games(definition, every=10):
                state = game.view_for(RED)
                legal = legal_actions(state)
                for choice in choices(legal):
                    if choice.amount != SURPLUS:
                        continue
                    action = decode_action(choice, legal)
                    offered += 1
                    self.assertLess(action.units, int(game.bases[action.source_id].units))
                    self.assertTrue(_kept(game, action, action.source_id), (definition.name, action))
                    more = SendAction(action.source_id, action.target_id, action.units + 1)
                    near_misses += not _kept(game, more, action.source_id)
        self.assertGreater(offered, 30)
        self.assertGreater(near_misses, 0.9 * offered)

    def test_teacher_moves_map_to_the_nearest_option(self):
        legal = legal_actions(Game(line_map()).view_for(RED))
        left, middle = row(legal, "left"), row(legal, "middle")
        # Options send 6 (just enough), then 3, 5, 8, 10, 13, 15, 18, 20.
        self.assertEqual(encode_action(SendAction("left", "middle", 6), legal), SendChoice(left, middle, JUST_ENOUGH))
        self.assertEqual(encode_action(SendAction("left", "middle", 11), legal).amount, AMOUNT_OPTIONS.index("50%"))
        self.assertEqual(encode_action(SendAction("left", "middle", 14), legal).amount, AMOUNT_OPTIONS.index("62.5%"))
        self.assertEqual(encode_action(SendAction("left", "middle", 20), legal).amount, ALL)

    def test_rejected_teacher_moves_count_as_waiting(self):
        legal = legal_actions(Game(line_map()).view_for(RED))
        self.assertIsNone(encode_action(SendAction("left", "middle", 21), legal))
        self.assertIsNone(encode_action(SendAction("left", "right", 5), legal))
        self.assertIsNone(encode_action(SendAction("right", "middle", 5), legal))
        self.assertIsNone(encode_action(SendAction("left", "nowhere", 5), legal))

    def test_encoding_a_decoded_choice_sends_the_same_units(self):
        for game in scripted_games(generate_map(6).definition, every=50):
            legal = legal_actions(game.view_for(BLUE))
            for choice in choices(legal):
                action = decode_action(choice, legal)
                self.assertEqual(decode_action(encode_action(action, legal), legal), action)

    def test_each_route_has_its_most_and_what_a_capture_needs(self):
        checked = 0
        for game in scripted_games(generate_map(6).definition, every=50):
            state = game.view_for(RED)
            legal = legal_actions(state)
            usable = legal.allowed.any(axis=-1)
            for source, target in zip(*np.nonzero(usable)):
                self.assertEqual(legal.most[source, target], int(state.bases[source].units))
                checked += 1
            self.assertFalse(legal.most[~usable].any())
            # Where "just enough" is on offer, it is exactly what a capture needs.
            offered = legal.allowed[..., JUST_ENOUGH]
            np.testing.assert_array_equal(legal.needed[offered], legal.units[..., JUST_ENOUGH][offered])
            self.assertTrue((legal.needed >= 0).all())
        self.assertGreater(checked, 20)

    def test_free_amounts_may_send_every_whole_number_up_to_most(self):
        game = scripted_game(generate_map(6).definition, 150)
        legal = legal_actions(game.view_for(RED))
        source, target = map(int, np.argwhere(legal.most == legal.most.max())[0])
        most = int(legal.most[source, target])
        self.assertGreater(most, 8)
        for units in range(1, most + 1):
            action = decode_action(SendChoice(source, target, units), legal, free=True)
            self.assertEqual(action.units, units)
            self.assertIsNone(game._validate_action(RED, action))
        for units in (0, most + 1):
            with self.assertRaises(ValueError):
                decode_action(SendChoice(source, target, units), legal, free=True)

    def test_teacher_moves_keep_their_units_with_free_amounts(self):
        legal = legal_actions(Game(line_map()).view_for(RED))
        left, middle = row(legal, "left"), row(legal, "middle")
        self.assertEqual(encode_action(SendAction("left", "middle", 11), legal, free=True), SendChoice(left, middle, 11))
        self.assertEqual(encode_action(SendAction("left", "middle", 20), legal, free=True), SendChoice(left, middle, 20))
        self.assertIsNone(encode_action(SendAction("left", "middle", 21), legal, free=True))
        self.assertIsNone(encode_action(SendAction("left", "right", 5), legal, free=True))


if __name__ == "__main__":
    unittest.main()
