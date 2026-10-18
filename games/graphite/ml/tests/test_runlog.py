import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ml.training.runlog
from ml.models.checkpoint import (
    history_entry,
    init_model,
    load_checkpoint,
    save_checkpoint,
)
from ml.play.players import min_send_gap
from ml.training.runlog import RunLog, read_metrics


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

    def test_a_run_folder_holds_its_settings_and_its_rounds_in_order(self):
        log = self.quiet_log()
        self.assertEqual(log.folder, self.root / "demo")
        self.assertEqual(json.loads((log.folder / "settings.json").read_text()), {"games": 5})
        self.assertEqual(read_metrics("demo"), [])
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

    def test_an_existing_run_is_resumed_or_replaced_only_when_asked(self):
        log = self.quiet_log()
        with mock.patch("builtins.print"):
            log.log(1, {"score/overall": 0.25}, "first")
        with self.assertRaises(FileExistsError):
            self.quiet_log()
        # Resuming keeps everything and keeps counting.
        (log.folder / "extra.txt").write_text("kept")
        with mock.patch("builtins.print"):
            resumed = RunLog("demo", {"games": 999}, resume=True)
            resumed.log(2, {"score/overall": 0.5}, "second")
        self.assertEqual([entry["round"] for entry in read_metrics("demo")], [1, 2])
        self.assertEqual((log.folder / "extra.txt").read_text(), "kept")
        self.assertEqual(json.loads((log.folder / "settings.json").read_text()), {"games": 5})
        with self.assertRaises(FileNotFoundError):
            RunLog("never_started", {}, resume=True)
        replaced = self.quiet_log(overwrite=True)
        self.assertEqual(read_metrics("demo"), [])
        self.assertTrue(replaced.folder.exists())

    def test_a_network_keeps_its_send_limit_wherever_it_is_copied(self):
        start = init_model("random", seed=1)
        limited = start.history + [history_entry("self-play", run="demo", min_send_gap=10)]
        self.quiet_log().save(1, start.model, limited)
        # The shared ml/runs/latest.pt has no run folder next to it, only its own history.
        self.assertEqual(min_send_gap(self.root / "latest.pt"), 10)
        # Training on without a limit lifts it; a network that never had one has none.
        lifted = limited + [history_entry("self-play", run="later")]
        self.assertEqual(min_send_gap(save_checkpoint(self.root / "lifted.pt", start.model, lifted)), 0)
        self.assertEqual(min_send_gap(save_checkpoint(self.root / "plain.pt", start.model, start.history)), 0)


if __name__ == "__main__":
    unittest.main()
