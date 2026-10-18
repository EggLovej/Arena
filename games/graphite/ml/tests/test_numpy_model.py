import os
import pickle
import subprocess
import sys
import tempfile
import unittest
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from game import Game
from game_types import BLUE, GREEN, RED
from loading import load_ai, load_map

from ml.encoding.action_encoder import decode_action
from ml.models import amounts, make_input
from ml.models.checkpoint import init_model, load_checkpoint, save_checkpoint
from ml.models.interface import ILLEGAL
from ml.play.numpy_model import NumpyModel, choose, on_grid, read_checkpoint
from ml.models.policy import to_choices
from ml.play.numpy_agent import NumpyAgent
from ml.tests.test_models import positions


ROOT = Path(__file__).resolve().parents[2]
COMMITTED = [ROOT / "checkpoints" / "neural_ai.pt", ROOT / "checkpoints" / "beast_ai.pt"]
# Architectures the numpy network must also follow, beyond the committed ones.
SETTINGS = [
    {},
    {"amounts": "free"},
    {"amounts": "free", "feature_set": "v2", "layers": 2},
    {"route_layers": 3, "width": 32, "heads": 2},
    {"feature_set": "v1"},
]


class ReadCheckpointTests(unittest.TestCase):
    def test_reads_every_committed_checkpoint_as_torch_does(self):
        for path in COMMITTED:
            with self.subTest(path.name):
                expected = torch.load(path, map_location="cpu", weights_only=True)
                contents = read_checkpoint(path)
                self.assertEqual(contents["model_config"], expected["model_config"])
                self.assertEqual(contents["encoding"], expected["encoding"])
                self.assertEqual(contents["weights"].keys(), expected["weights"].keys())
                for name, weight in expected["weights"].items():
                    np.testing.assert_array_equal(contents["weights"][name], weight.numpy(), err_msg=name)

    def test_refuses_a_file_that_holds_code(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "code.pt"
            torch.save({"format_version": 1, "danger": Game}, path)
            with self.assertRaises(Exception):
                read_checkpoint(path)

    def test_refuses_a_tensor_that_reaches_outside_its_numbers(self):
        def tensor_file(folder, offset, size, stride):
            storage = object()

            class Tensor:
                def __reduce__(self):
                    return torch._utils._rebuild_tensor_v2, (storage, offset, size, stride, False, OrderedDict())

            class Pickler(pickle.Pickler):
                def persistent_id(self, value):
                    return ("storage", torch.FloatStorage, "0", "cpu", 2) if value is storage else None

            path = Path(folder) / "tensor.pt"
            with zipfile.ZipFile(path, "w") as archive:
                with archive.open("tensor/data.pkl", "w") as file:
                    Pickler(file, protocol=2).dump({"format_version": 1, "weight": Tensor()})
                archive.writestr("tensor/byteorder", "little")
                archive.writestr("tensor/data/0", np.array([1.5, 2.5], dtype=np.float32).tobytes())
            return path

        with tempfile.TemporaryDirectory() as folder:
            np.testing.assert_array_equal(read_checkpoint(tensor_file(folder, 1, (1,), (1,)))["weight"], [2.5])
            for offset, size, stride in [(0, (3,), (1,)), (2, (1,), (1,)), (0, (2,), (5,)), (-1, (1,), (1,))]:
                with self.subTest(offset=offset, size=size, stride=stride), self.assertRaises(ValueError):
                    read_checkpoint(tensor_file(folder, offset, size, stride))


class NumpyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states, cls.legal = positions(seeds=range(4), every=50)

    def assert_same_scores(self, torch_model, model: NumpyModel):
        torch_model = torch_model.eval()
        for state, legal in zip(self.states, self.legal):
            with torch.no_grad():
                expected = torch_model(make_input([state], [legal]))
            output = model(state, legal)
            route = expected.route_logits[0].numpy()
            legal_routes = route > ILLEGAL / 2
            np.testing.assert_array_equal(output.route_logits > ILLEGAL / 2, legal_routes)
            np.testing.assert_allclose(output.route_logits[legal_routes], route[legal_routes], rtol=1e-4, atol=1e-4)
            self.assertAlmostEqual(output.value, float(expected.value[0]), places=4)
            usable = legal.allowed.any(axis=-1)
            if model.free_amounts:
                np.testing.assert_allclose(
                    output.amount_curves[usable], expected.amount_curves[0].numpy()[usable], rtol=1e-4, atol=1e-4
                )
            else:
                amount = expected.amount_logits[0].numpy()
                legal_amounts = amount > ILLEGAL / 2
                np.testing.assert_array_equal(output.amount_logits > ILLEGAL / 2, legal_amounts)
                np.testing.assert_allclose(output.amount_logits[legal_amounts], amount[legal_amounts], rtol=1e-4, atol=1e-4)

    def test_committed_networks_score_as_in_torch(self):
        for path in COMMITTED:
            with self.subTest(path.name):
                self.assert_same_scores(load_checkpoint(path).model, NumpyModel.from_file(path))

    def test_every_architecture_scores_as_in_torch(self):
        for number, settings in enumerate(SETTINGS):
            with self.subTest(**settings), tempfile.TemporaryDirectory() as folder:
                torch_model = init_model("random", seed=number, **settings).model
                # Untrained networks score every choice about the same; spread them out.
                with torch.no_grad():
                    for weight in torch_model.parameters():
                        weight.add_(torch.randn(weight.shape, generator=torch.Generator().manual_seed(number)) * 0.3)
                path = save_checkpoint(Path(folder) / "model.pt", torch_model)
                self.assert_same_scores(load_checkpoint(path).model, NumpyModel.from_file(path))

    def test_greedy_choices_are_the_torch_choices(self):
        for path in COMMITTED[:2]:
            torch_model, model = load_checkpoint(path).model.eval(), NumpyModel.from_file(path)
            for state, legal in zip(self.states, self.legal):
                with torch.no_grad():
                    expected = torch_model(make_input([state], [legal]))
                from ml.models.policy import choose as torch_choose

                route, amount = torch_choose(expected, greedy=True)
                wanted = to_choices(route, amount, bases=len(legal.base_ids))[0]
                chosen = choose(model(state, legal), greedy=True)
                self.assertEqual(
                    decode_action(chosen, legal, free=model.free_amounts),
                    decode_action(wanted, legal, free=model.free_amounts),
                )


class FreeAmountTests(unittest.TestCase):
    def test_amount_odds_are_the_torch_odds(self):
        generator = torch.Generator().manual_seed(0)
        for most, needed in ((1, 0), (2, 3), (7, 2), (40, 0), (60, 25)):
            parameters = torch.randn(amounts.CURVES, amounts.PARAMETERS, generator=generator) * 2
            expected = amounts.on_grid(parameters[None], torch.tensor([float(most)]), torch.tensor([float(needed)]), most)[0]
            np.testing.assert_allclose(on_grid(parameters.numpy(), most, needed, most), expected.numpy(), atol=1e-5)

    def test_sharpened_draws_follow_the_sharpened_odds(self):
        parameters = np.array([[0.0, 0.0, -6.0, 0.0, 1.5], [0.0, 6.0, -6.0, 1.0, 0.5], [-2.0, -6.0, 0.0, 0.0, 0.2]])
        most, needed = 12, 5
        bases = 2
        curves = np.zeros((bases, bases, amounts.CURVES, amounts.PARAMETERS), np.float32)
        curves[0, 1] = parameters
        from ml.play.numpy_model import NumpyOutput

        route_logits = np.full(1 + bases * bases, ILLEGAL, np.float32)
        route_logits[1 + 0 * bases + 1] = 0.0  # the only choice: from base 0 to base 1
        limits = np.zeros((bases, bases), np.float32)
        output = NumpyOutput(route_logits, None, 0.0, curves, limits + (np.arange(4).reshape(2, 2) == 1) * most, limits + needed)
        generator = np.random.default_rng(0)
        drawn = np.bincount([choose(output, generator=generator, temperature=0.5).amount for _ in range(4000)], minlength=most + 1)[1:]
        expected = on_grid(parameters, most, needed, most) ** 2
        expected /= expected.sum()
        np.testing.assert_allclose(drawn / drawn.sum(), expected, atol=0.03)


class NumpyAgentTests(unittest.TestCase):
    def test_plays_whole_games_without_a_refused_move(self):
        for path, map_name in ((COMMITTED[0], "crossroads"), (COMMITTED[1], "triangle")):
            with self.subTest(path.name):
                agent = NumpyAgent.from_init(path, seed=0, temperature=0.2)
                strategic = load_ai("strategic_ai")
                game = Game(load_map(map_name))
                players = {RED: agent, BLUE: strategic, GREEN: strategic}
                sends = 0
                while not game.is_finished() and game.tick < 600:
                    actions = {player: players[player](game.view_for(player)) for player in game.players}
                    sends += actions[RED] is not None
                    self.assertEqual(game.step(actions), {})
                self.assertGreater(sends, 10)

    def test_the_neural_bots_play_without_torch(self):
        # A fresh Python in which importing torch fails, as in the browser.
        script = """
import sys
class NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
sys.meta_path.insert(0, NoTorch())
from game import Game
from game_types import BLUE, RED
from loading import load_ai, load_map
strategic = load_ai("strategic_ai")
for name in ("neural_ai", "beast_ai"):
    bot = load_ai(name)
    game = Game(load_map("crossroads"))
    sends = 0
    while game.tick < 200 and not game.is_finished():
        action = bot(game.view_for(RED))
        sends += action is not None
        assert not game.step({RED: action, BLUE: strategic(game.view_for(BLUE))})
    assert sends > 5, (name, sends)
    assert "torch" not in sys.modules
print("ok")
"""
        environment = {**os.environ, "NEURAL_AI_CHECKPOINT": str(COMMITTED[0]), "BEAST_AI_FILE": str(ROOT / "checkpoints" / "beast.json")}
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
