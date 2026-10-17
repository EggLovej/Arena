import tempfile
import unittest
from pathlib import Path

from ml.evaluation.behaviour import measure, table
from ml.models.checkpoint import init_model, save_checkpoint


class BehaviourTests(unittest.TestCase):
    def test_a_network_is_measured_and_shown_in_a_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(save_checkpoint(Path(directory) / "net.pt", init_model("random", seed=0, layers=1).model))
            result = measure(path, games=2, workers=1)
        # An untrained network is undecided: its amounts are close to an even pick.
        self.assertGreater(result["amount uncertainty, share of an even pick"], 0.9)
        self.assertTrue(0 <= result["behaviour/sends to own bases"] <= 1)
        self.assertTrue(0 <= result["behaviour/units away from the front"] <= 1)
        self.assertEqual(result["send gap, ticks"], 0)
        lines = table({path: result}).splitlines()
        self.assertEqual(len(lines), 14)
        self.assertIn("sends to own bases", lines[6])

    def test_a_network_from_a_limited_run_is_measured_under_its_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "settings.json").write_text('{"min_send_gap": 10}')
            path = str(save_checkpoint(Path(directory) / "round_0000.pt", init_model("random", seed=0, layers=1).model))
            result = measure(path, games=2, workers=1)
        self.assertEqual(result["send gap, ticks"], 10)
        self.assertLessEqual(result["sends per second"], 1.0)
        self.assertGreater(result["units per send"], 0)

    def test_a_network_with_free_amounts_is_measured_over_every_whole_number(self):
        with tempfile.TemporaryDirectory() as directory:
            model = init_model("random", seed=0, layers=1, amounts="free", feature_set="v2").model
            path = str(save_checkpoint(Path(directory) / "free.pt", model))
            result = measure(path, games=2, workers=1)
        # Fresh curves sit on three amounts, so it is far from an even pick over every whole number.
        self.assertLess(result["amount uncertainty, share of an even pick"], 0.9)
        self.assertTrue(0 < result["favourite amount's probability"] <= 1)
        for name in ("hold taken when on offer", "surplus taken when on offer"):
            value = result[name]
            self.assertTrue(value != value or 0 <= value <= 1, name)


if __name__ == "__main__":
    unittest.main()
