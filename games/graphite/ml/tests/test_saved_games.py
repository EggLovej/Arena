import tempfile
import unittest
from pathlib import Path

from game import Game
from game_types import BLUE, RED
from main import play_headless

from ml.evaluation.evaluate import evaluate
from ml.evaluation.rematch import describe, rematch
from ml.evaluation.saved_games import find_game, list_games, load_game
from ml.evaluation.watch import recorded_players


class SavedGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # An untrained network loses most games to aggressive_ai: plenty to save.
        cls.directory = tempfile.TemporaryDirectory()
        cls.evaluation = evaluate(
            "untrained", ["aggressive_ai", "careful_ai"], generated_maps=3, bundled_maps=False,
            workers=1, max_seconds=60, save_losses_against=["aggressive_ai"], save_directory=cls.directory.name,
        )
        cls.saved = sorted(Path(cls.directory.name).rglob("*.json"))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_only_losses_against_the_listed_opponents_are_saved(self):
        expected = [r for r in self.evaluation.results if r.outcome == "loss" and r.plan.opponent == "aggressive_ai"]
        self.assertGreater(len(expected), 0)
        self.assertEqual(sorted(Path(r.saved_to) for r in expected), self.saved)
        self.assertTrue(all(not r.saved_to for r in self.evaluation.results if r not in expected))

    def test_a_saved_game_replays_to_the_same_end(self):
        for path in self.saved:
            saved = load_game(path)
            self.assertEqual((saved.candidate, saved.opponent, saved.outcome), ("untrained", "aggressive_ai", "loss"))
            game = saved.replay()
            self.assertEqual(game.winner(), -saved.candidate_seat)
            self.assertAlmostEqual(game.time_seconds, saved.seconds, places=1)

    def test_the_watch_window_plays_back_exactly_the_recorded_moves(self):
        saved = load_game(self.saved[0])
        players = recorded_players(saved)
        game = Game(saved.map)
        play_headless(game, players[RED], players[BLUE], round(saved.max_seconds / game.seconds_per_tick))
        replayed = saved.replay()
        self.assertEqual((game.winner(), game.tick), (replayed.winner(), replayed.tick))

    def test_saved_games_can_be_picked_by_number(self):
        listing = list_games(self.directory.name)
        for path in self.saved:
            self.assertIn(path.stem, listing)
        self.assertEqual(find_game("1", self.directory.name), self.saved[0])
        self.assertEqual(find_game(str(self.saved[-1])), self.saved[-1])
        self.assertIsNone(find_game(None))
        with self.assertRaises(SystemExit):
            find_game(str(len(self.saved) + 1), self.directory.name)

    def test_a_rematch_plays_both_seats_and_counts(self):
        saved = load_game(self.saved[0])
        results = rematch(saved, games=2, workers=1)
        self.assertEqual({seat: len(played) for seat, played in results.items()}, {"red": 2, "blue": 2})
        # aggressive_ai never picks by chance, and the same player against it plays the same game.
        same = rematch(saved, games=2, candidate="strategic_ai", workers=1)
        for played in same.values():
            self.assertEqual(played[0], played[1])
        table = describe(saved, results)
        self.assertIn("candidate plays", table)
        self.assertIn(saved.map_source, table)


if __name__ == "__main__":
    unittest.main()
