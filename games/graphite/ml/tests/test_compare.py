import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ml.training.runlog
from ml.evaluation.compare import compare, describe, fit_ratings
from ml.play.rollout import Seat
from ml.models.checkpoint import history_entry, init_model
from ml.training.runlog import RunLog


def fake_run(name, seconds_per_round, rounds=(0, 1, 2), earlier_run=None, seed=0, min_send_gap=0):
    """A run folder with a checkpoint and a logged time for each round."""

    log = RunLog(name, {})
    start = init_model("random", seed=seed)
    history = list(start.history)
    if earlier_run:
        history.append(history_entry("imitation", run=earlier_run))
    if min_send_gap:
        history.append(history_entry("self-play", run=name, min_send_gap=min_send_gap))
    for round_number in rounds:
        log.save(round_number, start.model, history)
    with (log.folder / "metrics.jsonl").open("w") as file:
        for round_number in rounds:
            file.write(json.dumps({"round": round_number, "seconds": round_number * seconds_per_round}) + "\n")
    return log.folder


class CompareTests(unittest.TestCase):
    """One comparison, read by every test: a run from random weights, one that started from an
    imitation run, and one trained under a send limit."""

    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.root = Path(directory.name)
        patches = (
            mock.patch.object(ml.training.runlog, "RUNS_DIRECTORY", cls.root),
            mock.patch("builtins.print"),
            mock.patch("sys.stdout", io.StringIO()),  # the live board of the matches
        )
        for patch in patches:
            patch.start()
            cls.addClassCleanup(patch.stop)

        teacher = RunLog("teacher", {})
        (teacher.folder / "metrics.jsonl").write_text(json.dumps({"round": 20, "seconds": 1800.0}) + "\n")
        cls.runs = {
            ("random start", "seed 0"): fake_run("a0", 36.0, rounds=(0, 1), seed=0),
            ("imitation start", "seed 0"): fake_run("b0", 36.0, rounds=(0,), earlier_run="teacher", seed=1),
            ("limited", "seed 0"): fake_run("limited", 10.0, rounds=(0, 1), seed=2, min_send_gap=10),
        }
        cls.limited_checkpoint = str(cls.runs[("limited", "seed 0")] / "round_0001.pt")
        # A player built on a checkpoint, whose name ends in .pt.
        cls.built_on_one = "network:" + str(cls.runs[("random start", "seed 0")] / "round_0000.pt")
        panel = ["strategic_ai", cls.limited_checkpoint, cls.built_on_one]
        with mock.patch("ml.evaluation.compare.Seat", wraps=Seat) as seat:
            cls.report = compare("demo", cls.runs, rounds=[0, 1, 50], panel=panel, games=2, workers=2)
        cls.seats = {(call.args[0], call.kwargs["min_send_gap"]) for call in seat.call_args_list}

    def test_checkpoints_of_all_runs_end_up_on_one_scale(self):
        rows = {(row["group"], row["round"]): row for row in self.report["checkpoints"]}
        # Rounds a run does not have are skipped.
        self.assertEqual(
            sorted(rows), [("imitation start", 0), ("limited", 0), ("limited", 1), ("random start", 0), ("random start", 1)]
        )
        # Time includes the imitation run that came first.
        self.assertAlmostEqual(rows[("random start", 1)]["hours"], 36 / 3600)
        self.assertAlmostEqual(rows[("imitation start", 0)]["hours"], 0.5)
        self.assertAlmostEqual(self.report["panel ratings"]["strategic_ai"], 1000.0, places=3)
        # The last checkpoints of every two groups met directly.
        self.assertEqual([match["games"] for match in self.report["head to head"]], [2, 2, 2])

        folder = self.root / "compare_demo"
        self.assertTrue((folder / "report.txt").exists())
        chart = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([entry["round"] for entry in chart], [0, 1])
        self.assertIn("rating/random start", chart[1])
        self.assertIn("rating/imitation start", chart[0])

    def test_a_run_trained_under_a_send_gap_plays_under_it(self):
        limited = {"limited/seed 0/round 0", "limited/seed 0/round 1", self.limited_checkpoint}
        self.assertEqual(self.report["send gaps"], {name: 10 for name in limited})
        self.assertEqual({player for player, gap in self.seats if gap == 10}, limited)
        self.assertIn(("random start/seed 0/round 1", 0), self.seats)
        self.assertIn(("strategic_ai", 0), self.seats)
        self.assertIn("Limited to one send every N ticks", describe(self.report))

        # --unlimited lifts every limit.
        runs = {("limited", "seed 0"): self.runs[("limited", "seed 0")]}
        with mock.patch("ml.evaluation.compare.Seat", wraps=Seat) as seat:
            report = compare("lifted", runs, rounds=[0], panel=["strategic_ai"], games=2, workers=2, unlimited=True)
        self.assertEqual(report["send gaps"], {})
        self.assertEqual({call.kwargs["min_send_gap"] for call in seat.call_args_list}, {0})
        self.assertIn("without a send limit", describe(report))

    def test_a_player_built_on_a_checkpoint_is_not_loaded_as_one(self):
        self.assertIn(self.built_on_one, self.report["panel ratings"])


class DescribeTests(unittest.TestCase):
    def test_groups_are_averaged_over_seeds_per_round(self):
        def row(group, label, round_number, rating, hours):
            return {"group": group, "label": label, "round": round_number, "hours": hours, "rating": rating, "against strategic_ai": 0.5}

        report = {
            "name": "demo", "rounds": [0, 100], "games per match": 100,
            "checkpoints": [
                row("A", "1", 0, 600, 0.0), row("A", "2", 0, 700, 0.0),
                row("A", "1", 100, 1000, 1.0), row("A", "2", 100, 1100, 1.2),
                row("B", "1", 0, 1000, 0.5), row("B", "1", 100, 1200, 1.5),
            ],
            "head to head": [{"first": "A/1/round 100", "second": "B/1/round 100", "score": 0.4, "games": 100}],
            "panel ratings": {"strategic_ai": 1000.0},
        }
        text = describe(report)
        self.assertIn("1050 (1000-1100) at 1.1 h", text)
        self.assertIn("650 (600-700) at 0.0 h", text)
        self.assertIn("A against B: 40.0% of 100 games (-2.0 standard errors from even)", text)


class RatingTests(unittest.TestCase):
    def test_the_ratings_explain_every_result_at_once(self):
        # 75% corresponds to about 191 points.
        ratings = fit_ratings({("a", "b"): [750.0, 1000]})
        self.assertAlmostEqual(ratings["a"] - ratings["b"], 191, delta=5)
        # "c" never met "a", but beat "b", who drew with "a": so c rates above a.
        ratings = fit_ratings({("a", "b"): [50.0, 100], ("c", "b"): [70.0, 100]})
        self.assertGreater(ratings["c"], ratings["a"])
        self.assertAlmostEqual(ratings["a"], ratings["b"], delta=15)

    def test_the_anchors_average_one_thousand_and_a_perfect_record_stays_finite(self):
        results = {("a", "b"): [70.0, 100], ("b", "c"): [60.0, 100], ("new", "a"): [80.0, 100]}
        ratings = fit_ratings(results, anchors=("a", "b", "c"))
        self.assertAlmostEqual((ratings["a"] + ratings["b"] + ratings["c"]) / 3, 1000, places=3)
        self.assertGreater(ratings["new"], ratings["a"])
        self.assertGreater(ratings["a"], ratings["b"])
        self.assertGreater(ratings["b"], ratings["c"])
        ratings = fit_ratings({("a", "b"): [20.0, 20]})
        self.assertTrue(400 < ratings["a"] - ratings["b"] < 2000)


if __name__ == "__main__":
    unittest.main()
