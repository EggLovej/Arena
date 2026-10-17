import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ml.evaluation.lineage as lineage
import ml.training.runlog
from ml.models.checkpoint import history_entry, init_model, save_checkpoint


class LineageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for patch in (
            mock.patch.object(ml.training.runlog, "RUNS_DIRECTORY", self.root),
            mock.patch.object(lineage, "RUNS_DIRECTORY", self.root),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def run_folder(self, name, rounds, entropy):
        folder = self.root / name
        folder.mkdir()
        (folder / "settings.json").write_text(json.dumps({"entropy_weight": entropy, "learning_rate": 1e-4}))
        lines = [{"round": r, "seconds": 10.0 * r, "score/vs starting model": 0.5, "entropy/of the policy": float(r)} for r in range(rounds + 1)]
        (folder / "metrics.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    def comparison(self, name, group, run, ratings):
        folder = self.root / f"compare_{name}"
        folder.mkdir()
        rows = [{"group": group, "label": "seed 0", "round": r, "rating": v, "against strategic_ai": 0.5} for r, v in ratings.items()]
        (folder / "results.json").write_text(json.dumps({"runs": {f"{group}/seed 0": f"ml/runs/{run}"}, "checkpoints": rows}))

    def test_runs_are_joined_round_after_round_and_ratings_chain_linked(self):
        self.run_folder("first", 4, 0.01)
        self.run_folder("second", 2, 0.001)
        self.comparison("one", "a", "first", {0: 900, 4: 1100})
        # A different panel: the same network that ended the first run rates 1050 here.
        self.comparison("two", "b", "second", {0: 1050, 2: 1150})
        history = [
            history_entry("random start"),
            history_entry("self-play", rounds=4, run="first"),
            history_entry("self-play", rounds=2, run="second"),
        ]
        checkpoint = save_checkpoint(self.root / "net.pt", init_model("random", seed=0, layers=1).model, history)

        segments, other = lineage.segments_of(str(checkpoint))
        self.assertEqual([(s.run, s.first, s.last) for s in segments], [("first", 0, 4), ("second", 4, 6)])
        rows, described = lineage.join(segments, {})
        self.assertEqual([row["round"] for row in rows], list(range(7)))
        # The second run's round 0 is the first run's round 4: not repeated, and its rounds follow on.
        self.assertEqual(rows[5]["entropy/of the policy"], 1.0)
        self.assertEqual(rows[5]["seconds"], 40.0 + 10.0)
        self.assertEqual(rows[5]["entropy bonus/weight"], 0.001)
        self.assertIn("score/vs the start of its run", rows[1])
        ratings = {row["round"]: row["rating/linked across comparisons"] for row in rows if "rating/linked across comparisons" in row}
        self.assertEqual(ratings, {0: 900, 4: 1100, 6: 1200})
        self.assertEqual(described[1]["rating at end, linked"], 1200)
        self.assertEqual(other[0]["step"], "random start")


if __name__ == "__main__":
    unittest.main()
