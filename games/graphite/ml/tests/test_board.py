import json
import tempfile
import unittest
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ml.training.board import Export
from ml.training.runlog import METRICS_FILE


def read_events(folder: Path) -> dict[str, list[tuple[int, float, float]]]:
    """Every chart in a TensorBoard folder: name -> (round, value, wall time) for each point."""

    events = EventAccumulator(str(folder), size_guidance={"scalars": 0})
    events.Reload()
    return {
        name: [(point.step, point.value, point.wall_time) for point in events.Scalars(name)]
        for name in events.Tags()["scalars"]
    }


class ExportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run_folder = Path(directory.name) / "my_run"
        self.run_folder.mkdir()
        self.target = Path(directory.name) / "tensorboard" / "my_run"
        self.export = Export(self.run_folder, self.target)
        self.addCleanup(self.export.close)

    def write(self, *lines: str, mode: str = "a") -> None:
        with (self.run_folder / METRICS_FILE).open(mode) as file:
            file.write("".join(lines))

    def test_every_metric_becomes_a_chart_by_round_and_training_time(self):
        self.write(
            json.dumps({"round": 0, "seconds": 16.0, "score/vs strategic_ai": 0.0}) + "\n",
            json.dumps({"round": 1, "seconds": 89.0, "entropy/of the policy": 3.47, "learning rate": 1e-4}) + "\n",
        )
        self.assertEqual(self.export.update(), 2)
        events = read_events(self.target)
        # round and seconds are the x-axes, not charts of their own.
        self.assertEqual(sorted(events), ["entropy/of the policy", "learning rate", "score/vs strategic_ai"])
        self.assertEqual(events["score/vs strategic_ai"], [(0, 0.0, 16.0)])
        [(step, value, wall_time)] = events["entropy/of the policy"]
        self.assertEqual((step, wall_time), (1, 89.0))
        self.assertAlmostEqual(value, 3.47, places=5)

    def test_values_that_are_not_numbers_are_left_out(self):
        self.write(json.dumps({"round": 1, "seconds": 1.0, "loss/value": None, "note/text": "x", "loss/move": 2.0}) + "\n")
        self.export.update()
        self.assertEqual(list(read_events(self.target)), ["loss/move"])

    def test_rounds_logged_later_are_added_and_a_half_written_line_waits(self):
        self.write(json.dumps({"round": 1, "seconds": 60.0, "loss/move": 2.0}) + "\n")
        self.export.update()
        line = json.dumps({"round": 2, "seconds": 120.0, "loss/move": 1.5}) + "\n"
        self.write(line[:10])
        self.assertEqual(self.export.update(), 0)
        self.write(line[10:])
        self.assertEqual(self.export.update(), 1)
        self.assertEqual(self.export.update(), 0)
        self.assertEqual(read_events(self.target)["loss/move"], [(1, 2.0, 60.0), (2, 1.5, 120.0)])

    def test_a_run_started_again_from_scratch_replaces_its_copy(self):
        self.write(*(json.dumps({"round": n, "seconds": 60.0 * n, "loss/move": 2.0}) + "\n" for n in (1, 2, 3)))
        self.export.update()
        self.write(json.dumps({"round": 1, "seconds": 50.0, "loss/move": 9.0}) + "\n", mode="w")
        self.assertEqual(self.export.update(), 1)
        self.assertEqual(read_events(self.target)["loss/move"], [(1, 9.0, 50.0)])


if __name__ == "__main__":
    unittest.main()
