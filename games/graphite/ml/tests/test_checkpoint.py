import tempfile
import unittest
from pathlib import Path

import torch

from game import Game
from game_types import RED

from ml.encoding.action_encoder import legal_actions
from ml.encoding.state_encoder import encode_state
from ml.map_generator import generate_map
from ml.models import build_model, make_input
from ml.models.checkpoint import (
    history_entry,
    init_model,
    load_checkpoint,
    model_settings,
    save_checkpoint,
)


def sample_input():
    states, legal = [], []
    for seed in range(4):
        state = Game(generate_map(seed).definition).view_for(RED)
        states.append(encode_state(state))
        legal.append(legal_actions(state))
    return make_input(states, legal)


def same_weights(first, second):
    return all(
        torch.equal(a, b)
        for a, b in zip(first.state_dict().values(), second.state_dict().values())
    )


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)

    def test_a_saved_model_comes_back_identical(self):
        inputs = sample_input()
        for settings in ({}, {"width": 32, "layers": 2}, {"amounts": "free", "feature_set": "v2"}):
            with self.subTest(**settings):
                start = init_model("random", seed=1, **settings)
                loaded = load_checkpoint(save_checkpoint(self.folder / "model.pt", start.model, start.history))
                self.assertEqual(loaded.model_name, "graph_transformer")
                self.assertEqual(loaded.model.config, start.model.config)
                self.assertTrue(same_weights(start.model, loaded.model))
                with torch.no_grad():
                    before, after = start.model(inputs), loaded.model(inputs)
                for name in ("route_logits", "amount_logits", "amount_curves", "value"):
                    if getattr(before, name) is not None:
                        self.assertTrue(torch.equal(getattr(before, name), getattr(after, name)), name)

    def test_a_seeded_random_start_repeats_and_leaves_other_randomness_alone(self):
        self.assertTrue(same_weights(init_model("random", seed=5).model, init_model("random", seed=5).model))
        self.assertFalse(same_weights(init_model("random", seed=5).model, init_model("random", seed=6).model))
        torch.manual_seed(123)
        expected = torch.rand(3)
        torch.manual_seed(123)
        init_model("random", seed=5)
        self.assertTrue(torch.equal(torch.rand(3), expected))

    def test_a_checkpoint_carries_on_with_its_history(self):
        start = init_model("random", seed=3)
        self.assertEqual([entry["step"] for entry in start.history], ["random start"])
        self.assertEqual(start.history[0]["seed"], 3)

        # What a trainer does: carry on from a checkpoint, add its own entry, save.
        first = save_checkpoint(self.folder / "a.pt", start.model, start.history)
        imitation = init_model(str(first))
        self.assertTrue(same_weights(start.model, imitation.model))
        imitation.history.append(history_entry("imitation", teacher="strategic_ai", games=100))
        second = save_checkpoint(self.folder / "b.pt", imitation.model, imitation.history)

        final = load_checkpoint(second)
        self.assertEqual([entry["step"] for entry in final.history], ["random start", "imitation"])
        self.assertEqual(final.history[1]["teacher"], "strategic_ai")
        self.assertEqual(len(load_checkpoint(first).history), 1)

    def test_training_state_comes_back(self):
        model = build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        sum(parameter.sum() for parameter in model.parameters()).backward()
        optimizer.step()

        state = {"optimizer": optimizer.state_dict(), "games_played": 12}
        path = save_checkpoint(self.folder / "train.pt", model, training_state=state)
        loaded = load_checkpoint(path)
        self.assertEqual(loaded.training_state["games_played"], 12)

        resumed = torch.optim.Adam(loaded.model.parameters(), lr=1e-3)
        resumed.load_state_dict(loaded.training_state["optimizer"])
        self.assertEqual(resumed.state_dict()["state"].keys(), optimizer.state_dict()["state"].keys())

    def test_saving_creates_folders_and_replaces_the_file_whole(self):
        path = save_checkpoint(self.folder / "runs" / "demo" / "model.pt", init_model("random", seed=1).model)
        second = init_model("random", seed=2).model
        save_checkpoint(path, second)
        self.assertTrue(same_weights(load_checkpoint(path).model, second))
        self.assertEqual([entry.name for entry in path.parent.iterdir()], ["model.pt"])

    def test_changed_encoders_are_refused(self):
        for table, change in (
            ("base_features", lambda names: names[:3] + ["something_else"] + names[4:]),
            ("global_features", lambda names: names[:3] + ["renamed"]),
        ):
            with self.subTest(table):
                path = save_checkpoint(self.folder / "model.pt", build_model())
                contents = torch.load(path, weights_only=True)
                contents["encoding"][table] = change(contents["encoding"][table])
                torch.save(contents, path)
                with self.assertRaisesRegex(ValueError, table):
                    load_checkpoint(path)

    def test_a_model_not_listed_in_ml_models_is_refused(self):
        path = save_checkpoint(self.folder / "model.pt", build_model())
        contents = torch.load(path, weights_only=True)
        contents["model_name"] = "retired_model"
        torch.save(contents, path)
        with self.assertRaisesRegex(ValueError, "retired_model"):
            load_checkpoint(path)

        class Unlisted(type(build_model())):
            pass

        with self.assertRaises(ValueError):
            save_checkpoint(self.folder / "unlisted.pt", Unlisted())

    def test_a_file_saved_before_a_setting_existed_or_with_one_since_removed_still_loads(self):
        start = init_model("random", seed=1)
        path = save_checkpoint(self.folder / "model.pt", start.model)
        contents = torch.load(path, weights_only=True)
        del contents["model_config"]["route_layers"]
        # The capture output, with its weights: it never steered a move, so it is left out.
        config = start.model.config
        contents["model_config"]["capture_head"] = True
        contents["weights"]["capture_out.weight"] = torch.zeros(config.amount_options, config.width)
        contents["weights"]["capture_out.bias"] = torch.zeros(config.amount_options)
        torch.save(contents, path)
        loaded = load_checkpoint(path)
        self.assertEqual(loaded.model.config.route_layers, 1)
        self.assertTrue(same_weights(start.model, loaded.model))

    def test_a_file_saved_before_features_were_added_still_loads(self):
        # As saved before the route features and the last global features existed.
        old = build_model(route_features=0, global_features=4)
        path = save_checkpoint(self.folder / "model.pt", old)
        contents = torch.load(path, weights_only=True)
        del contents["encoding"]["route_features"]
        del contents["model_config"]["route_features"]
        contents["encoding"]["global_features"] = contents["encoding"]["global_features"][:4]
        torch.save(contents, path)
        loaded = load_checkpoint(path).model
        self.assertEqual(loaded.config.route_features, 0)
        self.assertEqual(loaded.config.global_features, 4)
        self.assertTrue(same_weights(old, loaded))

    def test_settings_from_the_command_line_get_their_types(self):
        settings = model_settings("graph_transformer", ["width=128", "route_layers=2", "amounts=free"])
        self.assertEqual(settings, {"width": 128, "route_layers": 2, "amounts": "free"})
        self.assertIsInstance(settings["width"], int)
        for wrong in (["no_such_setting=3"], ["width"], ["width=wide"]):
            with self.assertRaises(SystemExit):
                model_settings("graph_transformer", wrong)


if __name__ == "__main__":
    unittest.main()
