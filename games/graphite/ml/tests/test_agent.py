import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ais.neural_ai
from game import Game
from game_types import BLUE, RED, BaseDefinition, EdgeDefinition, MapDefinition
from loading import available_ais, load_ai

from ml.map_generator import generate_map
from ml.models.checkpoint import init_model, save_checkpoint
from ml.models.policy import choose
from ml.play.agent import NeuralAgent


def play(definition, red, blue, ticks=300):
    """Play and record every move; fails if the game rejects any of them."""

    game = Game(definition)
    moves = []
    for _ in range(ticks):
        if game.is_finished():
            break
        actions = {RED: red(game.view_for(RED)), BLUE: blue(game.view_for(BLUE))}
        moves.append(actions[RED])
        rejected = game.step(actions)
        assert not rejected, rejected
    return game, moves


class NeuralAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = init_model("random", seed=0).model
        cls.strategic = staticmethod(load_ai("strategic_ai"))

    def test_untrained_agent_only_makes_legal_moves(self):
        for seed in range(4):
            agent = NeuralAgent(self.model, seed=seed)
            game, moves = play(generate_map(seed).definition, agent, self.strategic)
            self.assertGreater(sum(move is not None for move in moves), 10)

    def test_an_untrained_network_with_free_amounts_only_makes_legal_moves(self):
        model = init_model("random", seed=0, amounts="free").model
        for seed in range(2):
            agent = NeuralAgent(model, seed=seed)
            game, moves = play(generate_map(seed).definition, agent, self.strategic)  # fails on any rejected move
            self.assertGreater(sum(move is not None for move in moves), 10)

    def test_two_agents_can_share_one_model(self):
        red = NeuralAgent(self.model, seed=1)
        blue = NeuralAgent(self.model, seed=2)
        game, _ = play(generate_map(3).definition, red, blue, ticks=200)
        self.assertGreater(game.tick, 0)

    def test_same_seed_replays_the_same_game(self):
        definition = generate_map(2).definition
        _, first = play(definition, NeuralAgent(self.model, seed=7), self.strategic, ticks=150)
        _, second = play(definition, NeuralAgent(self.model, seed=7), self.strategic, ticks=150)
        _, other = play(definition, NeuralAgent(self.model, seed=8), self.strategic, ticks=150)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_reseeding_restarts_the_sequence(self):
        definition = generate_map(2).definition
        agent = NeuralAgent(self.model, seed=7)
        _, first = play(definition, agent, self.strategic, ticks=100)
        agent.reseed(7)
        _, second = play(definition, agent, self.strategic, ticks=100)
        self.assertEqual(first, second)

    def test_a_temperature_is_used_for_every_choice(self):
        with mock.patch("ml.play.agent.choose", wraps=choose) as spy:
            play(generate_map(2).definition, NeuralAgent(self.model, seed=7, temperature=0.8), self.strategic, ticks=30)
        self.assertTrue(spy.call_args_list)
        self.assertTrue(all(call.args[3] == 0.8 for call in spy.call_args_list))

    def test_greedy_agent_needs_no_seed_to_repeat(self):
        definition = generate_map(2).definition
        _, first = play(definition, NeuralAgent(self.model, greedy=True), self.strategic, ticks=100)
        _, second = play(definition, NeuralAgent(self.model, greedy=True), self.strategic, ticks=100)
        self.assertEqual(first, second)

    def test_waits_without_asking_the_model_when_nothing_can_be_sent(self):
        definition = MapDefinition(
            name="empty-handed",
            width=300,
            height=100,
            bases=(
                BaseDefinition("left", RED, 0, 1, 20, 50),
                BaseDefinition("right", BLUE, 5, 1, 280, 50),
            ),
            edges=(EdgeDefinition("left", "right", 1),),
        )
        agent = NeuralAgent(self.model)
        with mock.patch.object(agent, "model", side_effect=AssertionError("model was asked")):
            self.assertIsNone(agent.choose_action(Game(definition).view_for(RED)))

    def test_a_send_gap_spaces_out_the_sends_of_each_player_it_plays_for(self):
        agent = NeuralAgent.from_init("random", seed=0, min_send_gap=10)
        for _ in range(2):  # a second game starts with a clean slate
            sends = {RED: [], BLUE: []}
            game = Game(generate_map(0).definition)
            for _ in range(200):
                actions = {seat: agent(game.view_for(seat)) for seat in (RED, BLUE)}
                for seat, action in actions.items():
                    if action is not None:
                        sends[seat].append(game.tick)
                game.step(actions)
            for ticks in sends.values():
                self.assertLess(ticks[0], 10)
                self.assertGreater(len(ticks), 5)
                self.assertTrue(all(later - earlier >= 10 for earlier, later in zip(ticks, ticks[1:])))

    def test_builds_from_a_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(Path(directory) / "model.pt", self.model)
            definition = generate_map(2).definition
            _, saved = play(definition, NeuralAgent.from_init(path, seed=3), self.strategic, ticks=100)
        _, original = play(definition, NeuralAgent(self.model, seed=3), self.strategic, ticks=100)
        self.assertEqual(saved, original)

    def test_a_network_name_carries_its_temperature(self):
        from ml.evaluation.evaluate import load_player
        from ml.play.players import network_player

        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(Path(directory) / "model.pt", self.model)
            agent = load_player(f"network:{path}?temperature=0.2", seed=3)
            self.assertIsInstance(agent, NeuralAgent)
            self.assertEqual(agent.temperature, 0.2)
            self.assertEqual(network_player(f"network:{path}").temperature, 1.0)
            with self.assertRaises(ValueError):
                network_player(f"network:{path}?temperature=0")
            definition = generate_map(2).definition
            _, named = play(definition, agent, self.strategic, ticks=100)
        _, built = play(definition, NeuralAgent(self.model, seed=3, temperature=0.2), self.strategic, ticks=100)
        self.assertEqual(named, built)


class NeuralAiFileTests(unittest.TestCase):
    def setUp(self):
        ais.neural_ai._agent = None
        self.addCleanup(setattr, ais.neural_ai, "_agent", None)

    def test_it_plays_at_its_temperature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(Path(directory) / "latest.pt", init_model("random", seed=1).model)
            agent = ais.neural_ai._load_agent(path)
        self.assertEqual(agent.temperature, ais.neural_ai.TEMPERATURE)
        self.assertLess(ais.neural_ai.TEMPERATURE, 1.0)
        self.assertFalse(agent.greedy)

    def test_a_new_match_picks_up_a_newer_checkpoint(self):
        first = init_model("random", seed=1).model
        second = init_model("random", seed=2).model
        definition = generate_map(1).definition
        neural = load_ai("neural_ai")

        def loaded(model):
            current = ais.neural_ai._agent.model.state_dict()
            return all((current[name] == weight).all() for name, weight in model.state_dict().items())

        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(Path(directory) / "latest.pt", first)
            with mock.patch.dict(os.environ, {"NEURAL_AI_CHECKPOINT": str(path)}):
                game = Game(definition)
                neural(game.view_for(RED))
                self.assertTrue(loaded(first))

                # A trainer saves a newer model while the match is running.
                save_checkpoint(path, second)
                os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 5))
                game.step({})
                neural(game.view_for(RED))
                self.assertTrue(loaded(first), "weights must not change in the middle of a match")

                neural(Game(definition).view_for(RED))
                self.assertTrue(loaded(second))

    def test_the_game_can_find_and_load_it(self):
        self.assertIn("neural_ai", available_ais())
        self.assertTrue(callable(load_ai("neural_ai")))

    def test_plays_the_checkpoint_named_in_the_environment(self):
        model = init_model("random", seed=4).model
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(Path(directory) / "model.pt", model)
            with mock.patch.dict(os.environ, {"NEURAL_AI_CHECKPOINT": str(path)}):
                game, moves = play(generate_map(1).definition, load_ai("neural_ai"), load_ai("strategic_ai"), ticks=100)
        self.assertGreater(sum(move is not None for move in moves), 0)
        loaded = ais.neural_ai._agent.model.state_dict()
        self.assertTrue(all((loaded[name] == weight).all() for name, weight in model.state_dict().items()))

    def test_a_network_from_a_limited_run_plays_under_its_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "settings.json").write_text('{"min_send_gap": 10}')
            path = save_checkpoint(Path(directory) / "round_0001.pt", init_model("random", seed=4).model)
            with mock.patch.dict(os.environ, {"NEURAL_AI_CHECKPOINT": str(path)}):
                with contextlib.redirect_stderr(io.StringIO()):
                    play(generate_map(1).definition, load_ai("neural_ai"), load_ai("strategic_ai"), ticks=20)
        self.assertEqual(ais.neural_ai._agent.min_send_gap, 10)

    def test_plays_untrained_and_says_so_when_there_is_no_checkpoint(self):
        messages = io.StringIO()
        with mock.patch.dict(os.environ, {"NEURAL_AI_CHECKPOINT": "/nowhere/model.pt"}):
            with contextlib.redirect_stderr(messages):
                play(generate_map(1).definition, load_ai("neural_ai"), load_ai("strategic_ai"), ticks=50)
        self.assertIn("untrained", messages.getvalue())
        self.assertEqual(messages.getvalue().count("\n"), 1)


if __name__ == "__main__":
    unittest.main()
