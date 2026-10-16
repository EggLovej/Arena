import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from game import Game
from game_types import BLUE, RED
from loading import load_ai, load_map

from ml.encoding.action_encoder import legal_actions
from ml.encoding.state_encoder import encode_state, layout_for
from ml.map_generator import generate_map
from ml.models import amounts, make_input
from ml.models.checkpoint import init_model, save_checkpoint
from ml.play.lookahead import LookaheadPlayer, LookaheadSettings, choose_row, solve_matrix_game
from ml.play.players import network_player
from ml.play.rollout import Matchup, Seat, play_games
from ml.play.value_player import after_send, favourite_sends


class MatrixGameTests(unittest.TestCase):
    def test_matching_pennies_is_a_coin_toss(self):
        ours, theirs = solve_matrix_game(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        np.testing.assert_allclose(ours, [0.5, 0.5], atol=0.02)
        np.testing.assert_allclose(theirs, [0.5, 0.5], atol=0.02)

    def test_rock_paper_scissors_mixes_all_three_evenly(self):
        table = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
        ours, _ = solve_matrix_game(table)
        np.testing.assert_allclose(ours, [1 / 3] * 3, atol=0.02)

    def test_a_move_that_is_always_better_is_always_played(self):
        ours, _ = solve_matrix_game(np.array([[1.0, 2.0], [0.0, 1.0]]))
        self.assertGreater(ours[0], 0.98)

    def test_the_three_rules(self):
        table = np.array([[0.0, 0.0], [1.0, -5.0], [-1.0, 2.0]])
        # Worst case: only waiting never loses. Expected, the opponent nearly always plays the first column: row 1.
        self.assertEqual(choose_row(table, "worst", np.array([0.9, 0.1])), 0)
        self.assertEqual(choose_row(table, "expected", np.array([0.9, 0.1])), 1)
        self.assertIn(choose_row(table, "nash", np.array([0.9, 0.1])), (0, 1, 2))


class LookaheadPlayerTests(unittest.TestCase):
    def test_the_opponents_fleet_is_theirs(self):
        state = Game(load_map("classic")).view_for(RED)
        their_view = replace(state, player_id=BLUE)
        legal = legal_actions(their_view)
        (send,), _, _ = favourite_sends(init_model("random", seed=0).model, their_view, layout_for(state), None, legal, 1)
        after = after_send(state, send, owner=BLUE)
        (fleet,) = after.fleets
        self.assertEqual((fleet.owner, fleet.source_id), (BLUE, send.source_id))
        source = next(base for base in after.bases if base.id == send.source_id)
        self.assertEqual(source.owner, BLUE)

    def test_it_plays_legal_moves_and_judges_at_most_the_whole_table(self):
        player = LookaheadPlayer(init_model("random", seed=0).model, LookaheadSettings(mine=3, theirs=2, rule="nash"))
        opponent, game = load_ai("strategic_ai"), Game(generate_map(5).definition)
        decisions = 0
        for _ in range(30):
            state = game.view_for(RED)
            decisions += bool(legal_actions(state).allowed.any())
            self.assertFalse(game.step({RED: player(state), BLUE: opponent(game.view_for(BLUE))}))
        self.assertLessEqual(player.positions_judged, decisions * 4 * 3)
        self.assertGreater(player.positions_judged, decisions)

    def test_a_free_network_offers_its_likeliest_routes_at_their_likeliest_amounts(self):
        model = init_model("random", seed=0, amounts="free", feature_set="v2").model
        # A position with plenty of routes: strategic_ai against itself for 10 seconds.
        strategic, game = load_ai("strategic_ai"), Game(generate_map(5).definition)
        for _ in range(100):
            game.step({seat: strategic(game.view_for(seat)) for seat in (RED, BLUE)})
        state = game.view_for(RED)
        layout, legal = layout_for(state), legal_actions(state)
        sends, probabilities, _ = favourite_sends(model, state, layout, None, legal, 4)
        self.assertEqual(len({(send.source_id, send.target_id) for send in sends}), 4)
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
        with torch.no_grad():
            output = model(make_input([encode_state(state, layout)], [legal]))
        for send in sends:
            source, target = legal.base_ids.index(send.source_id), legal.base_ids.index(send.target_id)
            likeliest = amounts.most_likely(output.amount_curves[0, source, target], output.most[0, source, target], output.needed[0, source, target])
            self.assertEqual(send.units, int(likeliest))

        # Two amounts per route: the same number of sends, from fewer routes.
        sends, probabilities, _ = favourite_sends(model, state, layout, None, legal, 4, amounts_per_route=2)
        routes = [(send.source_id, send.target_id) for send in sends]
        self.assertEqual(len(sends), 4)
        self.assertEqual(len(set(sends)), 4)
        self.assertLess(len(set(routes)), 4)

    def test_a_free_network_plays_with_the_lookahead(self):
        model = init_model("random", seed=0, amounts="free", feature_set="v2").model
        player = LookaheadPlayer(model, LookaheadSettings(mine=3, theirs=2, rule="nash", amounts=2))
        opponent, game = load_ai("strategic_ai"), Game(generate_map(5).definition)
        for _ in range(30):
            self.assertFalse(game.step({RED: player(game.view_for(RED)), BLUE: opponent(game.view_for(BLUE))}))
        self.assertGreater(player.positions_judged, 0)

    def test_an_unknown_rule_is_refused(self):
        with self.assertRaises(ValueError):
            LookaheadPlayer(init_model("random", seed=0).model, LookaheadSettings(rule="hope"))

    def test_lookahead_players_load_by_name_everywhere(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(Path(directory) / "model.pt", init_model("random", seed=0).model)
            player = network_player(f"lookahead:{path}?mine=4&rule=worst")
            self.assertIsInstance(player, LookaheadPlayer)
            self.assertEqual((player.settings.mine, player.settings.theirs, player.settings.rule), (4, 8, "worst"))
            matchup = Matchup(generate_map(1).definition, Seat(f"lookahead:{path}?mine=2&theirs=2"), Seat("strategic_ai"))
            (record,) = play_games([matchup], max_seconds=3)
            self.assertGreater(record.ticks, 0)


if __name__ == "__main__":
    unittest.main()
