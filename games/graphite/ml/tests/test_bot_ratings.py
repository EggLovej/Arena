import tempfile
import unittest
from pathlib import Path

from ais.neural_ai import TEMPERATURE
from loading import available_ais
from ml.evaluation.bot_ratings import MENU, NEURAL_BY_CHANCE, OLD_BEAST, rate
from ml.models.checkpoint import init_model, save_checkpoint


class BotRatingTests(unittest.TestCase):
    def test_every_ai_in_the_menu_is_rated(self):
        self.assertEqual(set(MENU), set(available_ais()))

    def test_ratings_put_strategic_ai_at_1000_and_follow_the_recorded_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(Path(directory) / "net.pt", init_model("random", seed=0, layers=1).model)
            report = rate(games=2, workers=1, neural=str(path), beast=f"network:{path}?temperature=0.2")
        self.assertEqual(set(report["ratings"]), set(MENU))
        self.assertEqual(report["ratings"]["strategic_ai"], 1000)
        # The old Beast beat Atlas 49.5 points of 60, and Atlas beat Atlas Clean 26.5 of 32.
        self.assertGreater(report["others"][OLD_BEAST], report["ratings"]["atlas_ai"])
        self.assertGreater(report["ratings"]["atlas_ai"], report["ratings"]["atlas_clean_ai"])
        # Neural AI is rated at its temperature; its network drawing by chance, which the
        # recorded matches were played against, is rated beside it.
        self.assertIn(f"at temperature {TEMPERATURE}", report["rated as"]["neural_ai"])
        self.assertIn(NEURAL_BY_CHANCE, report["others"])
        self.assertIn("temperature=0.2", report["rated as"]["beast_ai"])


if __name__ == "__main__":
    unittest.main()
