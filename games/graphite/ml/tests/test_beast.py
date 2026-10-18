import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from game import Game
from game_types import BLUE, RED
from loading import load_ai

import ml.play.beast as beast
from ml.map_generator import generate_map


class BeastTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.file = Path(directory.name) / "beast.json"
        # Beast is whoever the file BEAST_AI_FILE names is about; the Arena adapter sets it when imported.
        patcher = mock.patch.dict(os.environ, {"BEAST_AI_FILE": str(self.file)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_new_beast_keeps_the_earlier_ones(self):
        beast.crown("careful_ai", "first", self.file)
        entry = beast.crown("strategic_ai", "beat careful_ai", self.file)
        self.assertEqual(entry["player"], "strategic_ai")
        self.assertEqual([old["player"] for old in entry["before"]], ["careful_ai"])
        self.assertEqual(json.loads(self.file.read_text()), entry)
        self.assertIn("before", beast.describe(entry))

    def test_a_name_that_cannot_be_played_is_refused(self):
        with self.assertRaises(ValueError):
            beast.crown("no_such_ai", "?", self.file)
        self.assertFalse(self.file.exists())

    def test_beast_ai_plays_whoever_is_beast_and_notices_a_change(self):
        import ais.beast_ai as beast_ai

        beast.crown("aggressive_ai", "test", self.file)
        game, opponent = Game(generate_map(4).definition), load_ai("careful_ai")
        with mock.patch.object(beast_ai, "_player", None):
            for _ in range(20):
                self.assertFalse(game.step({RED: beast_ai.choose_action(game.view_for(RED)), BLUE: opponent(game.view_for(BLUE))}))
            self.assertEqual(beast_ai._loaded["player"], "aggressive_ai")
            beast.crown("strategic_ai", "test", self.file)
            beast_ai.choose_action(Game(generate_map(4).definition).view_for(RED))  # a new match starts at tick 0
            self.assertEqual(beast_ai._loaded["player"], "strategic_ai")


if __name__ == "__main__":
    unittest.main()
