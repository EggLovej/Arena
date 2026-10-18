import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ais.neural_ai
from game import Game
from game_types import BLUE, RED, BaseDefinition, EdgeDefinition, MapDefinition, SendAction
from loading import load_ai

from ml.map_generator import generate_map
from ml.models.checkpoint import history_entry, init_model, save_checkpoint
from ml.play.agent import NeuralAgent
from ml.play.players import fresh_ai, keeps_memory, load_player, network_player


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

    def test_an_untrained_network_only_makes_legal_moves(self):
        for amounts, seeds in (("options", range(4)), ("free", range(2))):
            model = init_model("random", seed=0, amounts=amounts).model
            for seed in seeds:
                with self.subTest(amounts=amounts, seed=seed):
                    _, moves = play(generate_map(seed).definition, NeuralAgent(model, seed=seed), self.strategic)
                    self.assertGreater(sum(move is not None for move in moves), 10)

    def test_a_seed_replays_the_same_game_and_greedy_needs_none(self):
        def moves(agent):
            return play(generate_map(2).definition, agent, self.strategic, ticks=100)[1]

        agent = NeuralAgent(self.model, seed=7)
        first = moves(agent)
        agent.reseed(7)
        self.assertEqual(moves(agent), first)
        self.assertEqual(moves(NeuralAgent(self.model, seed=7)), first)
        self.assertNotEqual(moves(NeuralAgent(self.model, seed=8)), first)
        self.assertEqual(moves(NeuralAgent(self.model, greedy=True)), moves(NeuralAgent(self.model, greedy=True)))

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

    def test_players_load_by_name(self):
        definition = generate_map(2).definition
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(Path(directory) / "model.pt", self.model)
            # A network: name carries its settings, and the evaluator's --greedy reaches it.
            agent = load_player(f"network:{path}?temperature=0.2", seed=3)
            self.assertEqual(network_player(f"network:{path}").temperature, 1.0)
            self.assertTrue(load_player(f"network:{path}", greedy=True).greedy)
            with self.assertRaises(ValueError):
                network_player(f"network:{path}?temperature=0")
            _, named = play(definition, agent, self.strategic, ticks=100)
            # So are a checkpoint, an untrained network and an AI file.
            state = Game(definition).view_for(RED)
            for name in (str(path), "untrained", "untrained:3", "strategic_ai"):
                action = load_player(name, seed=0)(state)
                self.assertTrue(action is None or isinstance(action, SendAction))
        _, built = play(definition, NeuralAgent(self.model, seed=3, temperature=0.2), self.strategic, ticks=100)
        self.assertEqual(named, built)

    def test_an_ai_with_memory_gets_a_fresh_player_every_time(self):
        self.assertTrue(keeps_memory("atlas_clean_ai"))
        self.assertIsNot(fresh_ai("atlas_clean_ai"), fresh_ai("atlas_clean_ai"))
        self.assertIsNot(load_player("atlas_clean_ai"), load_player("atlas_clean_ai"))
        # Plain AIs are shared, and names that are not AI files have no memory to keep.
        self.assertFalse(keeps_memory("strategic_ai"))
        self.assertIs(fresh_ai("strategic_ai"), fresh_ai("strategic_ai"))
        self.assertFalse(keeps_memory("value:ml/runs/latest.pt"))


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
        start = init_model("random", seed=4)
        with tempfile.TemporaryDirectory() as directory:
            limited = start.history + [history_entry("self-play", min_send_gap=10)]
            path = save_checkpoint(Path(directory) / "latest.pt", start.model, limited)
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
