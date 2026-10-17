import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ml.training.runlog
from ml.evaluation.summary import summarise
from ml.models.checkpoint import init_model, save_checkpoint


class SummaryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.runs = Path(directory.name)
        patcher = mock.patch.object(ml.training.runlog, "RUNS_DIRECTORY", self.runs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_run(self, name, rounds, **model_settings):
        folder = self.runs / name
        folder.mkdir()
        lines = [json.dumps({"round": number, "seconds": 60.0 * number, **metrics}) for number, metrics in rounds]
        (folder / "metrics.jsonl").write_text("\n".join(lines) + "\n")
        start = init_model("random", seed=0, **model_settings)
        save_checkpoint(folder / "latest.pt", start.model)

    def test_one_column_per_run_with_size_and_last_round(self):
        self.make_run("small", [(1, {"agreement/all": 0.5}), (2, {"agreement/all": 0.75, "move loss/training": 0.4})])
        self.make_run("wide", [(1, {"agreement/all": 0.8})], width=128)
        with mock.patch("ml.evaluation.summary.RUNS_DIRECTORY", self.runs):
            table = summarise(["small", "wide"]).splitlines()
        rows = {line.split("  ")[0].strip(): line.split() for line in table[1:]}
        self.assertEqual(table[0].split(), ["small", "wide"])
        self.assertEqual(rows["weights"][-2:], ["120,730", "470,810"])
        self.assertEqual(rows["width"][-2:], ["64", "128"])
        self.assertEqual(rows["agreement/all"][-2:], ["0.750", "0.800"])
        self.assertEqual(rows["move loss/training"][-2:], ["0.400", "-"])
        self.assertEqual(rows["time"][-2:], ["2m00s", "1m00s"])

    def test_metrics_can_be_filtered_by_word(self):
        self.make_run("small", [(1, {"agreement/all": 0.5, "move loss/training": 0.4})])
        with mock.patch("ml.evaluation.summary.RUNS_DIRECTORY", self.runs):
            table = summarise(["small"], ["agreement"])
        self.assertIn("agreement/all", table)
        self.assertNotIn("move loss", table)


if __name__ == "__main__":
    unittest.main()
