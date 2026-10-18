import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import torch

import ml.training.runlog
from game import Game
from game_types import BLUE, RED
from loading import load_ai

import ml.judge.games as games
from ml.judge.games import GameSettings, SELF, _lines, generate, load_positions
from ml.judge.progress import ProgressBlock
from ml.judge.train import JudgeSettings, copy_matching_weights, train_judge
from ml.map_generator import generate_map
from ml.models import build_model
from ml.models.checkpoint import init_model, load_checkpoint, save_checkpoint
from ml.models.judge import GraphJudge
from ml.play.players import network_player
from ml.tests.support import trajectory_input


TINY_GAMES = GameSettings(
    player="untrained-placeholder", self_games=4, games_per_opponent=2, opponents=("aggressive_ai",),
    positions_per_seat=5, held_out_share=0.25, test_share=0.25, max_seconds=15, games_per_chunk=3, workers=2,
)


class JudgeModelTests(unittest.TestCase):
    def test_three_probabilities_that_add_up_to_one_and_a_value_from_them(self):
        judge = build_model("graph_judge", layers=2).eval()
        from ml.encoding.action_encoder import legal_actions
        from ml.encoding.state_encoder import encode_state
        from ml.models import make_input

        views = [Game(generate_map(seed).definition).view_for(RED) for seed in range(3)]
        inputs = make_input([encode_state(v) for v in views], [legal_actions(v) for v in views])
        with torch.no_grad():
            probabilities = judge.outcome_probabilities(inputs)
            value = judge(inputs).value
        self.assertEqual(probabilities.shape, (3, 3))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(3)))
        self.assertTrue(torch.allclose(value, probabilities[:, 0] - probabilities[:, 2]))

    def test_a_linear_judge_reads_plain_unit_counts(self):
        from ml.encoding.action_encoder import legal_actions
        from ml.encoding.state_encoder import BASE_FEATURES, encode_state
        from ml.models import make_input
        from ml.models.judge import UNIT_COLUMNS, _linear_units

        view = Game(generate_map(5).definition).view_for(RED)
        inputs = make_input([encode_state(view)], [legal_actions(view)])
        linear = _linear_units(inputs.base_features, UNIT_COLUMNS)
        units = BASE_FEATURES.index("units")
        rows = {base_id: row for row, base_id in enumerate(encode_state(view).base_ids)}
        for base in view.bases:
            self.assertAlmostEqual(linear[0, rows[base.id], units].item() * 250, base.units, places=2)
        # Everything else is left as it was.
        others = [c for c in range(len(BASE_FEATURES)) if c not in UNIT_COLUMNS]
        self.assertTrue(torch.equal(linear[..., others], inputs.base_features[..., others]))
        self.assertEqual(build_model("graph_judge", unit_scale="linear")(inputs).value.shape, (1,))

    def test_a_judge_saves_and_loads_like_any_model(self):
        judge = build_model("graph_judge", width=32, layers=2)
        with tempfile.TemporaryDirectory() as directory:
            loaded = load_checkpoint(save_checkpoint(Path(directory) / "judge.pt", judge)).model
        self.assertIsInstance(loaded, GraphJudge)
        self.assertEqual(loaded.config, judge.config)

    def test_a_playing_networks_attention_layers_can_start_a_judge(self):
        network, judge = build_model(layers=6), build_model("graph_judge")
        copied = copy_matching_weights(network, judge)
        self.assertGreater(copied, 0.8 * len(judge.state_dict()))
        self.assertTrue(torch.equal(judge.blocks[0].query_key_value.weight, network.blocks[0].query_key_value.weight))


class PlayersWithAJudgeTests(unittest.TestCase):
    def test_the_judge_judges_and_the_network_proposes(self):
        with tempfile.TemporaryDirectory() as directory:
            network = save_checkpoint(Path(directory) / "network.pt", init_model("random", seed=0).model)
            judge = save_checkpoint(Path(directory) / "judge.pt", build_model("graph_judge", width=32, layers=2))
            lookahead = network_player(f"lookahead:{network}?rule=nash&judge={judge}")
            scorer = network_player(f"value:{network}?candidates=4&judge={judge}")
        for player in (lookahead, scorer):
            self.assertIsInstance(player.judge, GraphJudge)
            self.assertNotIsInstance(player.model, GraphJudge)
        game, opponent = Game(generate_map(3).definition), load_ai("strategic_ai")
        for _ in range(10):
            self.assertFalse(game.step({RED: lookahead(game.view_for(RED)), BLUE: opponent(game.view_for(BLUE))}))


class GamesAndTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.patches = [
            mock.patch.object(games, "DATA_DIRECTORY", root / "judge_data"),
            mock.patch.object(ml.training.runlog, "RUNS_DIRECTORY", root),
        ]
        for patch in cls.patches:
            patch.start()
        cls.network = str(save_checkpoint(root / "network.pt", init_model("random", seed=0).model))
        settings = GameSettings(**{**TINY_GAMES.__dict__, "player": cls.network})
        with redirect_stdout(io.StringIO()):
            cls.folder = generate("tiny", settings)
            cls.second = generate("tiny2", GameSettings(**{**settings.__dict__, "seed": 1}))

    @classmethod
    def tearDownClass(cls):
        for patch in cls.patches:
            patch.stop()
        cls.directory.cleanup()

    def test_every_game_is_played_and_its_positions_kept_in_one_split(self):
        state = json.loads((self.folder / "progress.json").read_text())
        self.assertEqual(state["games"], {SELF: 4, "aggressive_ai": 2})
        splits = {split: load_positions("tiny", split) for split in ("training", "held_out", "test")}
        total = sum(len(t) for trajectories in splits.values() for t in trajectories)
        self.assertEqual(total, sum(state["positions"].values()))
        for trajectories in splits.values():
            for trajectory in trajectories:
                self.assertLessEqual(len(trajectory), 5)
                self.assertIn(trajectory.outcome, ("win", "draw", "loss"))
                trajectory_input(trajectory)  # the kept positions still make model inputs
        # Both seats of a game are recorded, the scripted AI's too.
        players = {t.player for trajectories in splits.values() for t in trajectories}
        self.assertEqual(players, {self.network, "aggressive_ai"})

    def test_a_second_data_run_with_the_same_seed_is_refused(self):
        settings = GameSettings(**{**TINY_GAMES.__dict__, "player": self.network})
        with self.assertRaisesRegex(ValueError, "same maps"):
            generate("again", settings)

    def test_the_summary_gives_total_time_and_size(self):
        text = games.summary(self.folder)
        self.assertIn("Total: 6 games", text)
        self.assertIn("MB on disk", text)
        self.assertIn("of playing", text)

    def test_a_finished_run_cannot_be_overwritten_and_resuming_it_adds_nothing(self):
        with self.assertRaises(FileExistsError):
            generate("tiny", GameSettings(**{**TINY_GAMES.__dict__, "player": self.network}))
        with redirect_stdout(io.StringIO()):
            generate("tiny", resume=True)
        state = json.loads((self.folder / "progress.json").read_text())
        self.assertEqual(state["games"], {SELF: 4, "aggressive_ai": 2})

    def test_the_progress_shows_each_group_on_its_own_line(self):
        state = {
            "games": {SELF: 4, "aggressive_ai": 1, "careful_ai": 0},
            "seconds": {SELF: 125.0, "aggressive_ai": 10.0, "careful_ai": 0.0},
            "positions": {"training": 80, "held_out": 10, "test": 10},
        }
        targets = {SELF: 4, "aggressive_ai": 3, "careful_ai": 2}
        lines = _lines(state, targets, "aggressive_ai", {"games": 1, "positions": {"training": 7, "held_out": 0, "test": 0}})
        self.assertIn("took 2m05s", lines[0])
        self.assertIn("(2 / 3) vs Aggressive AI", lines[1])
        self.assertIn("5s remaining", lines[1])  # 10 s for 2 games, 1 still to play
        self.assertNotIn("remaining", lines[2])
        self.assertIn("positions recorded: 107", lines[3])

    def test_training_stops_once_the_held_out_loss_stops_improving(self):
        losses = iter([0.5, 0.4, 0.45, 0.46, 0.47, 0.3, 0.2])  # the 6th would be better, but it never comes
        settings = JudgeSettings(epochs=7, batch_size=16, patience=3)
        with redirect_stdout(io.StringIO()) as output, mock.patch("ml.judge.train._loss", lambda *_: next(losses)):
            best = train_judge("stopping", ["tiny", "tiny2"], settings, architecture={"width": 32, "layers": 2}, device="cpu")
        self.assertEqual(len(ml.training.runlog.read_metrics("stopping")), 5)
        self.assertIn("Stopping early", output.getvalue())
        self.assertIn("best: epoch 2", output.getvalue())
        self.assertEqual(load_checkpoint(best).history[-1]["epochs"], 2)

    def test_several_data_runs_are_trained_on_together(self):
        with redirect_stdout(io.StringIO()) as output:
            train_judge("both", ["tiny", "tiny2"], JudgeSettings(epochs=1, batch_size=16),
                        architecture={"width": 32, "layers": 2}, device="cpu")
        both = sum(len(t) for name in ("tiny", "tiny2") for t in load_positions(name, "training"))
        self.assertIn(f"training {both:,}", output.getvalue())

    def test_training_keeps_the_best_epoch_and_reports_on_the_test_games(self):
        settings = JudgeSettings(epochs=2, batch_size=16)
        with redirect_stdout(io.StringIO()):
            best = train_judge("judge", "tiny", settings, architecture={"width": 32, "layers": 2},
                               start_from=None, compare_with=self.network, device="cpu")
        self.assertIsInstance(load_checkpoint(best).model, GraphJudge)
        report = json.loads(best.with_name("test.json").read_text())
        for key in ("test loss", "score error", "score error, reference", "calibration"):
            self.assertIn(key, report)
        self.assertEqual(len(ml.training.runlog.read_metrics("judge")), 2)


class ProgressTests(unittest.TestCase):
    def test_in_a_log_the_block_is_printed_whole(self):
        stream = io.StringIO()
        block = ProgressBlock("Working", stream=stream, log_every=0)
        block.show(["(1 / 2) a", "(0 / 2) b"])
        block.finish(["(2 / 2) a", "(2 / 2) b"])
        text = stream.getvalue()
        self.assertIn("=== Working ===", text)
        self.assertIn("(2 / 2) b", text)
        self.assertNotIn("\x1b", text)


if __name__ == "__main__":
    unittest.main()
