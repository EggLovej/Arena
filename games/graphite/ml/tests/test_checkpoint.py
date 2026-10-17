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

    def test_loaded_model_gives_identical_answers(self):
        start = init_model("random", seed=1)
        path = save_checkpoint(self.folder / "model.pt", start.model, start.history)
        loaded = load_checkpoint(path)

        inputs = sample_input()
        with torch.no_grad():
            before = start.model(inputs)
            after = loaded.model(inputs)
        self.assertTrue(torch.equal(before.route_logits, after.route_logits))
        self.assertTrue(torch.equal(before.amount_logits, after.amount_logits))
        self.assertTrue(torch.equal(before.value, after.value))
        self.assertEqual(loaded.model_name, "graph_transformer")

    def test_a_network_with_free_amounts_and_a_feature_set_comes_back_the_same(self):
        start = init_model("random", seed=1, amounts="free", feature_set="v2")
        path = save_checkpoint(self.folder / "free.pt", start.model, start.history)
        loaded = load_checkpoint(path)
        self.assertEqual(loaded.model.config, start.model.config)
        inputs = sample_input()
        with torch.no_grad():
            before, after = start.model(inputs), loaded.model(inputs)
        self.assertTrue(torch.equal(before.route_logits, after.route_logits))
        self.assertTrue(torch.equal(before.amount_curves, after.amount_curves))
        self.assertEqual(
            model_settings("graph_transformer", ["amounts=free", "feature_set=v2"]), {"amounts": "free", "feature_set": "v2"}
        )

    def test_settings_are_restored(self):
        start = init_model("random", seed=1, width=32, layers=2)
        path = save_checkpoint(self.folder / "small.pt", start.model)
        loaded = load_checkpoint(path)
        self.assertEqual(loaded.model.config, start.model.config)
        self.assertEqual(loaded.model.config.width, 32)
        self.assertTrue(same_weights(start.model, loaded.model))

    def test_random_start_is_repeatable_with_a_seed(self):
        self.assertTrue(same_weights(init_model("random", seed=5).model, init_model("random", seed=5).model))
        self.assertFalse(same_weights(init_model("random", seed=5).model, init_model("random", seed=6).model))

    def test_seeded_start_leaves_other_randomness_alone(self):
        torch.manual_seed(123)
        expected = torch.rand(3)
        torch.manual_seed(123)
        init_model("random", seed=5)
        self.assertTrue(torch.equal(torch.rand(3), expected))

    def test_init_from_a_path_continues_that_model(self):
        start = init_model("random", seed=2)
        path = save_checkpoint(self.folder / "start.pt", start.model, start.history)
        continued = init_model(path)
        self.assertTrue(same_weights(start.model, continued.model))
        self.assertEqual(continued.history, start.history)
        self.assertTrue(same_weights(init_model(str(path)).model, start.model))

    def test_history_grows_step_by_step(self):
        start = init_model("random", seed=3)
        self.assertEqual([entry["step"] for entry in start.history], ["random start"])
        self.assertEqual(start.history[0]["seed"], 3)

        # What a trainer does: carry on from a checkpoint, add its own entry, save.
        first = save_checkpoint(self.folder / "a.pt", start.model, start.history)
        imitation = init_model(first)
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

    def test_saving_creates_folders_and_leaves_no_temporary_file(self):
        path = save_checkpoint(self.folder / "runs" / "demo" / "model.pt", build_model())
        self.assertTrue(path.exists())
        self.assertEqual([entry.name for entry in path.parent.iterdir()], ["model.pt"])

    def test_saving_again_replaces_the_file(self):
        path = self.folder / "model.pt"
        save_checkpoint(path, init_model("random", seed=1).model)
        second = init_model("random", seed=2).model
        save_checkpoint(path, second)
        self.assertTrue(same_weights(load_checkpoint(path).model, second))

    def test_missing_file_is_reported_clearly(self):
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(self.folder / "nothing.pt")

    def test_changed_encoders_are_refused(self):
        path = save_checkpoint(self.folder / "model.pt", build_model())
        contents = torch.load(path, weights_only=True)
        contents["encoding"]["base_features"][3] = "something_else"
        torch.save(contents, path)
        with self.assertRaisesRegex(ValueError, "base_features"):
            load_checkpoint(path)

    def test_unknown_architecture_is_refused(self):
        path = save_checkpoint(self.folder / "model.pt", build_model())
        contents = torch.load(path, weights_only=True)
        contents["model_name"] = "retired_model"
        torch.save(contents, path)
        with self.assertRaisesRegex(ValueError, "retired_model"):
            load_checkpoint(path)

    def test_a_file_saved_before_a_setting_existed_still_loads(self):
        start = init_model("random", seed=1)
        path = save_checkpoint(self.folder / "model.pt", start.model)
        contents = torch.load(path, weights_only=True)
        del contents["model_config"]["route_layers"]
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

    def test_a_feature_changed_in_place_is_still_refused(self):
        path = save_checkpoint(self.folder / "model.pt", build_model())
        contents = torch.load(path, weights_only=True)
        contents["encoding"]["global_features"] = contents["encoding"]["global_features"][:3] + ["renamed"]
        torch.save(contents, path)
        with self.assertRaisesRegex(ValueError, "global_features"):
            load_checkpoint(path)

    def test_settings_from_the_command_line_get_their_types(self):
        settings = model_settings("graph_transformer", ["width=128", "route_layers=2"])
        self.assertEqual(settings, {"width": 128, "route_layers": 2})
        self.assertIsInstance(settings["width"], int)
        for wrong in (["no_such_setting=3"], ["width"], ["width=wide"]):
            with self.assertRaises(SystemExit):
                model_settings("graph_transformer", wrong)

    def test_unregistered_model_cannot_be_saved(self):
        class Unlisted(type(build_model())):
            pass

        with self.assertRaises(ValueError):
            save_checkpoint(self.folder / "model.pt", Unlisted())


if __name__ == "__main__":
    unittest.main()
