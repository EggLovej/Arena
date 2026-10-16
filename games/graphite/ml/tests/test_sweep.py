import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ml.training.runlog
import ml.evaluation.sweep
from ml.evaluation.sweep import _convert, describe, parse_args, run_sweep
from ml.models.checkpoint import init_model
from ml.training.runlog import RunLog
from ml.training.selfplay import SelfPlaySettings


def fake_trainer(calls):
    """Writes what a finished run leaves behind, without training anything."""

    def train(run_name, settings, init="random", overwrite=False):
        calls.append((run_name, settings))
        log = RunLog(run_name, settings, overwrite)
        checkpoint = init_model("random", seed=settings.seed)
        log.save(settings.rounds, checkpoint.model, checkpoint.history)
        log.log(settings.rounds, {"entropy/of the policy": 4.0 + settings.entropy_weight}, "")

    return train


class SweepTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.calls = []
        patches = (
            mock.patch.object(ml.training.runlog, "RUNS_DIRECTORY", self.root),
            mock.patch.dict(ml.evaluation.sweep.TRAINERS, {"fake": (SelfPlaySettings, fake_trainer(self.calls))}),
            mock.patch("builtins.print"),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def sweep(self, variations, seeds=(0, 1)):
        fixed = dict(rounds=3, max_seconds=15.0)
        return run_sweep("demo", "fake", "random", fixed, variations, list(seeds), match_games=2, workers=2)

    def test_every_variation_is_trained_with_every_seed_and_rated(self):
        report = self.sweep([{"entropy_weight": 0.0}, {"entropy_weight": 0.02}])

        trained = [(name, settings.entropy_weight, settings.seed, settings.rounds) for name, settings in self.calls]
        self.assertEqual(
            trained,
            [
                ("sweep_demo/entropy_weight_0.0_seed0", 0.0, 0, 3),
                ("sweep_demo/entropy_weight_0.0_seed1", 0.0, 1, 3),
                ("sweep_demo/entropy_weight_0.02_seed0", 0.02, 0, 3),
                ("sweep_demo/entropy_weight_0.02_seed1", 0.02, 1, 3),
            ],
        )
        self.assertEqual(len(report["runs"]), 4)
        for row in report["runs"]:
            self.assertTrue(500 < row["rating"] < 1500)
            for opponent in ("the starting model", "strategic_ai", "careful_ai", "aggressive_ai"):
                self.assertTrue(0 <= row[f"against {opponent}"] <= 1)
            self.assertAlmostEqual(row["entropy at the end"], 4.0 + row["settings"]["entropy_weight"])
        self.assertEqual(set(report["ratings"]), {"starting model", "aggressive_ai", "careful_ai", "strategic_ai"})

        # Runs meet the starting model and the scripted AIs, and runs of the other variation, but not their own twin.
        met = {(first, second) for first, second, _, _ in report["results"]}
        first, twin, other = (row["run"] for row in report["runs"][:3])
        self.assertIn((first, "starting model"), met)
        self.assertIn((first, other), met)
        self.assertNotIn((first, twin), met)

        folder = self.root / "sweep_demo"
        self.assertTrue((folder / "report.txt").exists())
        self.assertEqual(json.loads((folder / "results.json").read_text())["name"], "demo")
        # A sweep must not replace the model the game plays.
        self.assertFalse((self.root / "latest.pt").exists())

    def test_finished_runs_are_not_trained_again(self):
        self.sweep([{"entropy_weight": 0.0}], seeds=[0])
        self.assertEqual(len(self.calls), 1)
        self.sweep([{"entropy_weight": 0.0}, {"entropy_weight": 0.02}], seeds=[0])
        self.assertEqual([name for name, _ in self.calls][1:], ["sweep_demo/entropy_weight_0.02_seed0"])

    def test_the_same_settings_under_two_names_are_trained_once(self):
        default = SelfPlaySettings().entropy_weight
        self.sweep([{"entropy_weight": default}, {"passes": SelfPlaySettings().passes}], seeds=[0])
        self.assertEqual(len(self.calls), 1)

    def test_a_failed_run_does_not_stop_the_sweep(self):
        working = ml.evaluation.sweep.TRAINERS["fake"][1]

        def sometimes_failing(run_name, settings, **kwargs):
            if settings.seed == 0:
                raise RuntimeError("this one breaks")
            working(run_name, settings, **kwargs)

        with mock.patch.dict(ml.evaluation.sweep.TRAINERS, {"fake": (SelfPlaySettings, sometimes_failing)}):
            report = self.sweep([{"entropy_weight": 0.0}])
        self.assertEqual([row["seed"] for row in report["runs"]], [1])


class CommandTests(unittest.TestCase):
    def test_a_separate_process_gets_exactly_the_same_settings(self):
        from ml.evaluation.sweep import _command
        from ml.training.train import parse_args as parse_train_args

        settings = SelfPlaySettings(rounds=7, entropy_weight=0.003, opponents=("self", "careful_ai"), final_evaluation=False)
        command = _command("selfplay", "sweep_x/a_seed0", "ml/runs/x.pt", settings)
        with mock.patch.object(sys, "argv", ["train", *command[command.index("selfplay") :]]):
            args = parse_train_args()
        rebuilt = SelfPlaySettings(
            **{
                name: tuple(value) if isinstance(value := getattr(args, name), list) else value
                for name in SelfPlaySettings.__dataclass_fields__
            }
        )
        self.assertEqual(rebuilt, settings)
        self.assertEqual((args.run, args.init, args.overwrite), ("sweep_x/a_seed0", "ml/runs/x.pt", True))


class ReportTests(unittest.TestCase):
    def test_variations_are_listed_best_first_with_their_spread(self):
        def row(variation, seed, rating):
            return {
                "variation": variation, "seed": seed, "run": f"{variation}_{seed}", "rating": rating,
                "against the starting model": 0.6, "against strategic_ai": 0.5, "against careful_ai": 0.9,
                "against aggressive_ai": 0.9, "entropy at the end": 1.0, "minutes": 12.0,
            }

        report = {
            "name": "demo", "trainer": "selfplay", "init": "x.pt", "games per match": 240,
            "runs": [row("a=1", 0, 1200), row("a=1", 1, 1260), row("a=2", 0, 1300), row("a=2", 1, 1340)],
            "ratings": {"strategic_ai": 1300.0, "starting model": 1250.0},
        }
        text = describe(report)
        self.assertLess(text.index("a=2"), text.index("a=1"))
        self.assertIn("1300-1340", text)
        self.assertIn("1200-1260", text)
        self.assertIn("1230", text)  # the mean of the weaker variation
        self.assertIn("strategic_ai 1300", text)


class CommandLineTests(unittest.TestCase):
    def test_values_take_the_type_of_the_setting(self):
        self.assertEqual(_convert(SelfPlaySettings, "rounds", "20"), 20)
        self.assertEqual(_convert(SelfPlaySettings, "entropy_weight", "0.003"), 0.003)
        self.assertIs(_convert(SelfPlaySettings, "final_evaluation", "no"), False)
        self.assertEqual(_convert(SelfPlaySettings, "opponents", "self+strategic_ai"), ("self", "strategic_ai"))
        with self.assertRaises(SystemExit):
            _convert(SelfPlaySettings, "no_such_setting", "1")

    def test_vary_and_variant_are_both_understood(self):
        arguments = [
            "sweep", "--name", "x", "--seeds", "0", "1", "2", "--set", "rounds=20",
            "--vary", "entropy_weight=0,0.01", "--variant", "clip=0.1", "passes=8",
        ]
        with mock.patch.object(sys, "argv", arguments):
            args = parse_args()
        self.assertEqual(args.seeds, [0, 1, 2])
        self.assertEqual(args.vary, ["entropy_weight=0,0.01"])
        self.assertEqual(args.variant, [["clip=0.1", "passes=8"]])


if __name__ == "__main__":
    unittest.main()
