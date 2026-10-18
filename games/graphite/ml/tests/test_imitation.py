import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import ml.training.runlog
from ml.models.checkpoint import init_model, load_checkpoint, save_checkpoint
from ml.map_generator import generate_map
from ml.play.rollout import Matchup, Seat, play_games
from ml.training.imitation import ImitationSettings, split_decisions, train_imitation
from ml.training.runlog import read_metrics
from ml.training.selfplay import SelfPlaySettings
from ml.training.train import parse_args


class ImitationTests(unittest.TestCase):
    def test_a_small_run_learns_and_leaves_a_complete_run_folder(self):
        settings = ImitationSettings(
            games=12,
            decisions_per_game=150,
            epochs=3,
            batch_size=128,
            evaluation_maps=0,
            final_evaluation=False,
            workers=2,
            validation_share=0.25,
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(ml.training.runlog, "RUNS_DIRECTORY", Path(directory)), mock.patch("builtins.print"):
                best = train_imitation("tiny", settings, device="cpu")
                rounds = read_metrics("tiny")
            folder = Path(directory) / "tiny"
            names = sorted(entry.name for entry in folder.iterdir())
            checkpoint = load_checkpoint(best)

        self.assertEqual(
            names,
            [
                "best.pt", "latest.pt", "metrics.jsonl", "round_0000.pt", "round_0001.pt",
                "round_0002.pt", "round_0003.pt", "settings.json",
            ],
        )
        # Round 0 is the starting model, measured before any training.
        self.assertEqual([entry["round"] for entry in rounds], [0, 1, 2, 3])
        self.assertNotIn("move loss/training", rounds[0])

        # The network gets better at predicting the teacher, on games it trained on and on others.
        self.assertLess(rounds[-1]["move loss/training"], rounds[1]["move loss/training"])
        self.assertLess(rounds[-1]["move loss/validation"], rounds[0]["move loss/validation"])
        # The reference line does not depend on the network, so it never moves.
        self.assertEqual(len({entry["value loss/always saying 0"] for entry in rounds}), 1)
        for entry in rounds:
            for name in ("all decisions", "when teacher sends", "when teacher waits"):
                self.assertGreaterEqual(entry[f"agreement/{name}"], 0.0)
                self.assertLessEqual(entry[f"agreement/{name}"], 1.0)

        # The checkpoint remembers how it was made.
        steps = [entry["step"] for entry in checkpoint.history]
        self.assertEqual(steps, ["random start", "imitation"])
        self.assertEqual(checkpoint.history[-1]["teacher"], "strategic_ai")
        self.assertEqual(checkpoint.history[-1]["epochs"], 3)

    def test_a_network_with_free_amounts_learns_the_teachers_units(self):
        settings = ImitationSettings(
            games=8, decisions_per_game=150, epochs=2, batch_size=128, evaluation_maps=0,
            final_evaluation=False, workers=2, validation_share=0.25,
        )
        with tempfile.TemporaryDirectory() as directory:
            start = save_checkpoint(Path(directory) / "free.pt", init_model("random", seed=0, amounts="free", feature_set="v2").model)
            with mock.patch.object(ml.training.runlog, "RUNS_DIRECTORY", Path(directory)), mock.patch("builtins.print"):
                best = train_imitation("free", settings, init=str(start), device="cpu")
                rounds = read_metrics("free")
            self.assertTrue(load_checkpoint(best).model.free_amounts)
        self.assertLess(rounds[-1]["move loss/validation"], rounds[0]["move loss/validation"])
        self.assertTrue(0 <= rounds[-1]["agreement/when teacher sends"] <= 1)


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        matchups = [
            Matchup(generate_map(seed).definition, Seat("strategic_ai", True), Seat("aggressive_ai"))
            for seed in range(12)
        ]
        cls.games = [record.trajectories for record in play_games(matchups, max_seconds=10)]

    def split(self, games, share):
        return split_decisions(games, 300, share, np.random.default_rng(0))

    def test_every_tenth_game_is_held_out_and_at_least_one(self):
        training, validation = self.split(self.games, 0.1)
        self.assertEqual(len(validation), sum(len(trajectory) for trajectory in self.games[9]))
        self.assertEqual(len(training) + len(validation), sum(len(t) for game in self.games for t in game))
        # Too few games for the share still hold one out; no share holds out none.
        training, validation = self.split(self.games[:4], 0.1)
        self.assertEqual(len(validation), sum(len(trajectory) for trajectory in self.games[3]))
        self.assertGreater(len(training), 0)
        self.assertEqual(len(self.split(self.games[:4], 0.0)[1]), 0)


class TrainCommandTests(unittest.TestCase):
    def test_every_setting_of_a_trainer_is_an_option_with_its_default(self):
        def parse(*arguments):
            with mock.patch.object(sys, "argv", ["train", *arguments]):
                return parse_args()

        args = parse("imitation", "--run", "x")
        defaults = ImitationSettings()
        self.assertEqual((args.run, args.init), ("x", "random"))
        self.assertEqual((args.games, args.learning_rate), (defaults.games, defaults.learning_rate))
        self.assertEqual(tuple(args.opponents), defaults.opponents)

        args = parse(
            "imitation", "--run", "x", "--games", "50", "--learning-rate", "0.01",
            "--opponents", "careful_ai", "aggressive_ai", "--init", "some/model.pt",
        )
        self.assertEqual((args.games, args.learning_rate, args.init), (50, 0.01, "some/model.pt"))
        self.assertEqual(args.opponents, ["careful_ai", "aggressive_ai"])

        args = parse("selfplay", "--run", "x", "--rounds", "7", "--opponents", "self", "careful_ai")
        self.assertEqual((args.trainer, args.rounds, args.opponents), ("selfplay", 7, ["self", "careful_ai"]))
        self.assertEqual(args.learning_rate, SelfPlaySettings().learning_rate)


if __name__ == "__main__":
    unittest.main()
