import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ml.training.runlog
from ml.models.checkpoint import init_model, load_checkpoint
from ml.training.runlog import RunLog, format_duration, min_send_gap, read_metrics


class RunLogTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        patcher = mock.patch.object(ml.training.runlog, "RUNS_DIRECTORY", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def quiet_log(self, name="demo", **kwargs):
        with mock.patch("builtins.print"):
            return RunLog(name, {"games": 5}, **kwargs)

    def test_creates_the_run_folder_with_its_settings(self):
        log = self.quiet_log()
        self.assertEqual(log.folder, self.root / "demo")
        self.assertEqual(json.loads((log.folder / "settings.json").read_text()), {"games": 5})
        self.assertEqual(read_metrics("demo"), [])

    def test_rounds_are_appended_and_read_back_in_order(self):
        log = self.quiet_log()
        with mock.patch("builtins.print") as printed:
            log.log(1, {"score/overall": 0.25}, "first")
            log.log(2, {"score/overall": 0.5}, "second")
        rounds = read_metrics("demo")
        self.assertEqual([entry["round"] for entry in rounds], [1, 2])
        self.assertEqual(rounds[1]["score/overall"], 0.5)
        self.assertIn("seconds", rounds[0])
        self.assertIn("second", printed.call_args.args[0])
        self.assertEqual(read_metrics(log.folder), rounds)

    def test_a_half_written_line_is_skipped(self):
        log = self.quiet_log()
        with mock.patch("builtins.print"):
            log.log(1, {"score/overall": 0.25}, "first")
        with (log.folder / "metrics.jsonl").open("a") as file:
            file.write('{"round": 2, "score/ov')
        self.assertEqual(len(read_metrics("demo")), 1)

    def test_an_existing_run_is_not_replaced_by_accident(self):
        log = self.quiet_log()
        with mock.patch("builtins.print"):
            log.log(1, {"score/overall": 0.25}, "first")
        with self.assertRaises(FileExistsError):
            self.quiet_log()
        replaced = self.quiet_log(overwrite=True)
        self.assertEqual(read_metrics("demo"), [])
        self.assertTrue(replaced.folder.exists())

    def test_saving_writes_the_round_the_latest_and_the_shared_latest(self):
        log = self.quiet_log()
        first = init_model("random", seed=1)
        second = init_model("random", seed=2)
        path = log.save(1, first.model, first.history, {"epoch": 1})
        log.mark_best(path)
        log.save(2, second.model, second.history, {"epoch": 2})

        names = sorted(entry.name for entry in log.folder.iterdir())
        self.assertEqual(names, ["best.pt", "latest.pt", "metrics.jsonl", "round_0001.pt", "round_0002.pt", "settings.json"])
        self.assertEqual(load_checkpoint(log.folder / "latest.pt").training_state, {"epoch": 2})
        self.assertIsNone(load_checkpoint(log.folder / "round_0002.pt").training_state)

        def weights(file):
            return load_checkpoint(file).model.state_dict()["embed_bases.weight"]

        self.assertTrue((weights(self.root / "latest.pt") == weights(log.folder / "round_0002.pt")).all())
        self.assertTrue((weights(log.folder / "best.pt") == weights(log.folder / "round_0001.pt")).all())
        self.assertFalse((weights(log.folder / "best.pt") == weights(log.folder / "latest.pt")).all())

    def test_resuming_keeps_everything_and_keeps_counting(self):
        log = self.quiet_log()
        with mock.patch("builtins.print"):
            log.log(1, {"score/overall": 0.25}, "first")
        (log.folder / "extra.txt").write_text("kept")
        with mock.patch("builtins.print"):
            resumed = RunLog("demo", {"games": 999}, resume=True)
            resumed.log(2, {"score/overall": 0.5}, "second")
        self.assertEqual([entry["round"] for entry in read_metrics("demo")], [1, 2])
        self.assertEqual((log.folder / "extra.txt").read_text(), "kept")
        self.assertEqual(json.loads((log.folder / "settings.json").read_text()), {"games": 5})
        with self.assertRaises(FileNotFoundError):
            RunLog("never_started", {}, resume=True)

    def test_a_checkpoint_has_the_send_gap_of_its_run(self):
        with mock.patch("builtins.print"):
            limited = RunLog("limited", {"min_send_gap": 10})
        self.assertEqual(min_send_gap(limited.folder / "round_0000.pt"), 10)
        self.assertEqual(min_send_gap(self.quiet_log().folder / "round_0003.pt"), 0)  # a run from before the setting
        self.assertEqual(min_send_gap(self.root / "loose.pt"), 0)  # not in a run folder

    def test_durations_read_naturally(self):
        self.assertEqual(format_duration(5), "5s")
        self.assertEqual(format_duration(75), "1m15s")
        self.assertEqual(format_duration(3 * 3600 + 5 * 60), "3h05m")


if __name__ == "__main__":
    unittest.main()
