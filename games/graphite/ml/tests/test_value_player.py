import tempfile
import unittest
from pathlib import Path

import torch

from game import Game
from game_types import BLUE, RED
from loading import load_ai, load_map

from ml.encoding.action_encoder import AMOUNT_OPTIONS, JUST_ENOUGH, SendChoice, decode_action, legal_actions
from ml.encoding.state_encoder import BASE_FEATURES, encode_state
from ml.map_generator import generate_map
from ml.models.checkpoint import init_model, save_checkpoint
from ml.models.interface import ModelInput, PolicyValueModel, Scores
from ml.play.players import load_player, network_player
from ml.play.rollout import Matchup, Seat, play_games
from ml.play.value_player import ValuePlayer, ValueSettings, after_send


FUTURE_IS_MINE = BASE_FEATURES.index("future_is_mine")


class OwnershipJudge(PolicyValueModel):
    """A stand-in value output: the more bases will be mine once every fleet has landed, the better."""

    def __init__(self, sign: float = 1.0):
        super().__init__()
        self.sign = sign
        self.level = torch.nn.Parameter(torch.zeros(()))

    def scores(self, inputs: ModelInput) -> Scores:
        batch, bases = inputs.base_mask.shape
        owned = (inputs.base_features[..., FUTURE_IS_MINE] * inputs.base_mask).sum(dim=1)
        return Scores(
            wait=self.level.expand(batch),
            route=self.level.expand(batch, bases, bases),
            amount=self.level.expand(batch, bases, bases, len(AMOUNT_OPTIONS)),
            value=self.sign * owned / 10,
        )


def capture_on_classic():
    """Red's opening position on `classic`, and a send that takes a neutral base with just enough units."""

    state = Game(load_map("classic")).view_for(RED)
    legal = legal_actions(state)
    source, target = next(zip(*legal.allowed[:, :, JUST_ENOUGH].nonzero()))
    return state, decode_action(SendChoice(int(source), int(target), JUST_ENOUGH), legal)


class AfterSendTests(unittest.TestCase):
    def test_the_units_leave_and_a_fleet_is_on_its_way(self):
        state, send = capture_on_classic()
        after = after_send(state, send)
        source = next(base for base in state.bases if base.id == send.source_id)
        self.assertAlmostEqual(next(b for b in after.bases if b.id == send.source_id).units, source.units - send.units)
        (fleet,) = set(after.fleets) - set(state.fleets)
        travel = next(e.travel_time for e in state.edges if {e.a, e.b} == {send.source_id, send.target_id})
        self.assertEqual(
            (fleet.owner, fleet.units, fleet.source_id, fleet.target_id, fleet.remaining_time), (RED, send.units, send.source_id, send.target_id, travel)
        )
        # The position it was made from is unchanged.
        self.assertEqual(state, Game(load_map("classic")).view_for(RED))

    def test_the_network_sees_the_capture_the_moment_the_fleet_leaves(self):
        state, send = capture_on_classic()
        before, after = encode_state(state), encode_state(after_send(state, send))
        row = before.base_ids.index(send.target_id)
        self.assertEqual(before.base_features[row, FUTURE_IS_MINE], 0.0)
        self.assertEqual(after.base_features[row, FUTURE_IS_MINE], 1.0)


class ValuePlayerTests(unittest.TestCase):
    def test_it_plays_the_move_its_judge_likes_best(self):
        state, _ = capture_on_classic()
        greedy = ValuePlayer(OwnershipJudge(sign=1.0))
        send = greedy(state)
        row = encode_state(state).base_ids.index(send.target_id)
        self.assertEqual(encode_state(after_send(state, send)).base_features[row, FUTURE_IS_MINE], 1.0)
        # A judge that dislikes owning bases keeps everything at home.
        self.assertIsNone(ValuePlayer(OwnershipJudge(sign=-1.0))(state))

    def test_candidates_limit_the_moves_it_looks_at(self):
        state, _ = capture_on_classic()
        everything, favourites = ValuePlayer(OwnershipJudge()), ValuePlayer(OwnershipJudge(), ValueSettings(candidates=2))
        everything(state)
        favourites(state)
        self.assertEqual(favourites.positions_judged, 3)  # waiting plus the two favourites
        self.assertGreater(everything.positions_judged, 3)
        # Every distinct send once: amount options that come to the same units are judged once.
        legal = legal_actions(state)
        distinct = {(s, t, int(legal.units[s, t, o])) for s, t, o in zip(*legal.allowed.nonzero())}
        self.assertEqual(everything.positions_judged, 1 + len(distinct))

    def test_an_untrained_network_plays_legal_moves(self):
        player, opponent = ValuePlayer(init_model("random", seed=0).model), load_ai("strategic_ai")
        game = Game(generate_map(5).definition)
        for _ in range(40):
            action = player(game.view_for(RED))
            self.assertFalse(game.step({RED: action, BLUE: opponent(game.view_for(BLUE))}))


class ValueNameTests(unittest.TestCase):
    def test_value_players_load_by_name_everywhere(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(Path(directory) / "model.pt", init_model("random", seed=0).model)
            player = network_player(f"value:{path}?candidates=5")
            self.assertIsInstance(player, ValuePlayer)
            self.assertEqual(player.settings.candidates, 5)
            self.assertIsInstance(load_player(f"value:{path}"), ValuePlayer)
            with self.assertRaises(ValueError):
                network_player(f"value:{path}?horizon=3")

            matchup = Matchup(generate_map(1).definition, Seat(f"value:{path}"), Seat("strategic_ai"))
            (record,) = play_games([matchup], max_seconds=3)
            self.assertGreater(record.ticks, 0)


if __name__ == "__main__":
    unittest.main()
