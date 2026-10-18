import unittest
from dataclasses import replace

import numpy as np
import torch

from game import Game
from game_types import BLUE, RED, SendAction
from loading import load_ai

from ml.encoding.action_encoder import SendChoice, decode_action, encode_action, legal_actions, option_units
from ml.encoding.state_encoder import encode_state
from ml.evaluation.evaluate import play_match
from ml.map_generator import generate_map
from ml.models.checkpoint import init_model
from ml.models.interface import ILLEGAL
from ml.models.policy import log_probability
from ml.play.rollout import (
    WAIT,
    Matchup,
    RolloutPool,
    Seat,
    count_decisions,
    play_games,
)
from ml.play.inference import InferenceServer
from ml.tests.support import repeated, replay, trajectory_choices, trajectory_input


def setUpModule():
    # One pool for the whole file, as starting one takes seconds. It plays the networks on a
    # server here in the main process (on the CPU), and scripted games in its workers.
    global pool
    pool = RolloutPool(workers=2, inference_device="cpu")


def tearDownModule():
    pool.close()


def matchups(red, blue, seeds=range(4), record=True):
    return [
        Matchup(generate_map(seed).definition, Seat(red, record), Seat(blue, record))
        for seed in seeds
    ]


def same_trajectories(first, second):
    return all(
        np.array_equal(getattr(a, name), getattr(b, name))
        for one, other in zip(first, second)
        for a, b in zip(one.trajectories, other.trajectories)
        for name in ("base_features", "allowed_bits", "source", "target", "amount", "tick")
    )


class ServerTests(unittest.TestCase):
    """Workers that leave the networks to one server in the main process (here on the CPU)."""

    @classmethod
    def setUpClass(cls):
        cls.models = {"net": init_model("random", seed=0).model, "other": init_model("random", seed=1).model}
        played = matchups("net", "net", seeds=range(3)) + matchups("net", "other", seeds=range(3, 6))
        cls.finished = []
        cls.records = pool.play(played, cls.models, seed=3, max_seconds=20, on_finished=lambda *args: cls.finished.append(args))

    def test_every_game_is_played_and_reported(self):
        self.assertEqual(len(self.records), 6)
        self.assertEqual(sorted(index for index, _, _ in self.finished), list(range(6)))
        for index, decisions, winner in self.finished:
            self.assertEqual(decisions, [len(t) for t in self.records[index].trajectories])
            self.assertEqual(winner, self.records[index].winner)

    def test_recorded_probabilities_and_values_are_what_each_network_says(self):
        players = set()
        for record in self.records:
            for trajectory in record.trajectories:
                players.add(trajectory.player)
                model = self.models[trajectory.player].eval()
                with torch.no_grad():
                    output = model(trajectory_input(trajectory))
                    expected = log_probability(output, *trajectory_choices(trajectory))
                self.assertTrue(torch.allclose(expected, torch.from_numpy(trajectory.log_probability), atol=1e-4))
                self.assertTrue(torch.allclose(output.value, torch.from_numpy(trajectory.value), atol=1e-4))
        self.assertEqual(players, {"net", "other"})

    def test_replaying_gives_the_same_game(self):
        for record in self.records:
            game = replay(record)
            self.assertEqual((game.winner(), game.tick), (record.winner, record.ticks))


class FreeAmountTests(unittest.TestCase):
    """A network with free amounts, played here and through the server, next to one with options."""

    @classmethod
    def setUpClass(cls):
        cls.models = {
            "free": init_model("random", seed=0, amounts="free", feature_set="v2").model,
            "options": init_model("random", seed=1).model,
        }
        cls.local = play_games(matchups("free", "strategic_ai", seeds=range(3)), cls.models, seed=2, max_seconds=30)
        cls.served = pool.play(matchups("free", "options", seeds=range(3, 7)), cls.models, seed=3, max_seconds=20)

    def trajectories(self):
        return [trajectory for record in self.local + self.served for trajectory in record.trajectories]

    def test_choices_are_units_from_one_to_everything_and_say_so(self):
        sends = set()
        for trajectory in self.trajectories():
            self.assertEqual(trajectory.free_amounts, trajectory.player == "free")
            sent = np.flatnonzero(trajectory.source != WAIT)
            most = trajectory.limit_table(sent)[np.arange(len(sent)), trajectory.source[sent], trajectory.target[sent], 0]
            self.assertTrue(((trajectory.units[sent] >= 1) & (trajectory.units[sent] <= most)).all())
            self.assertTrue((trajectory.units[trajectory.source == WAIT] == WAIT).all())
            if trajectory.free_amounts:
                np.testing.assert_array_equal(trajectory.amount[sent], trajectory.units[sent])
                sends.update(zip(trajectory.units[sent].tolist(), most.tolist()))
        # Not only the options' amounts: somewhere it sends neither 1 nor everything.
        self.assertTrue(any(1 < units < most for units, most in sends))

    def test_recorded_probabilities_and_values_are_what_each_network_says(self):
        checked = {"free": 0, "options": 0}
        for trajectory in self.trajectories():
            if trajectory.player not in self.models:
                continue
            with torch.no_grad():
                output = self.models[trajectory.player].eval()(trajectory_input(trajectory))
                expected = log_probability(output, *trajectory_choices(trajectory))
            np.testing.assert_allclose(expected.numpy(), trajectory.log_probability, atol=1e-4)
            np.testing.assert_allclose(output.value.numpy(), trajectory.value, atol=1e-4)
            checked[trajectory.player] += len(trajectory)
        self.assertGreater(min(checked.values()), 100)

    def test_every_move_is_accepted_and_the_games_replay(self):
        for record in self.local + self.served:
            self.assertEqual(replay(record).winner(), record.winner)


class RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = init_model("random", seed=0).model
        cls.models = {"net": cls.model}
        # One short self-play batch and one network-against-teacher batch, shared by several tests.
        cls.self_play = play_games(matchups("net", "net"), cls.models, seed=1, max_seconds=30)
        cls.mixed = play_games(matchups("net", "strategic_ai"), cls.models, seed=1, max_seconds=30)

    def test_scripted_games_match_the_plain_game_loop(self):
        records = play_games(matchups("strategic_ai", "aggressive_ai", record=False), max_seconds=60)
        for record in records:
            reference = play_match(record.matchup.map, load_ai("strategic_ai"), load_ai("aggressive_ai"), 60)
            self.assertEqual((record.winner, record.ticks), (reference.winner(), reference.tick))
            self.assertEqual(record.trajectories, [])

    def test_an_ai_with_memory_plays_each_game_as_if_alone(self):
        # The same map and seat twice at once, against different opponents.
        definition = generate_map(2).definition
        pairings = [Matchup(definition, Seat("atlas_clean_ai"), Seat(opponent)) for opponent in ("strategic_ai", "aggressive_ai")]
        together = play_games(pairings, max_seconds=4)
        for matchup, record in zip(pairings, together):
            alone = play_games([matchup], max_seconds=4)[0]
            self.assertEqual(record.actions, alone.actions)

    def test_every_move_is_accepted_and_replaying_them_gives_the_identical_game(self):
        for records in (self.self_play, self.mixed):
            for record in records:
                again = replay(record)
                self.assertEqual(again.tick, record.ticks)
                self.assertEqual(again.winner(), record.winner)
                self.assertEqual(again.is_finished(), record.winner is not None or again.living_players() == ())

    def test_recorded_decisions_match_the_positions_they_were_made_in(self):
        checked = 0
        for record in self.self_play[:2] + self.mixed[:2]:
            moves = {}
            for tick, seat, source, target, units in record.actions:
                moves.setdefault(tick, {})[seat] = SendAction(source, target, units)
            game = Game(record.matchup.map)
            pointer = {id(t): 0 for t in record.trajectories}

            while game.tick < record.ticks:
                for trajectory in record.trajectories:
                    at = pointer[id(trajectory)]
                    if at >= len(trajectory) or trajectory.tick[at] != game.tick:
                        continue
                    state = game.view_for(trajectory.seat)
                    legal = legal_actions(state)
                    np.testing.assert_array_equal(trajectory.allowed(slice(at, at + 1))[0], legal.allowed)
                    np.testing.assert_array_equal(trajectory.base_features[at], encode_state(state).base_features)
                    # The route limits rebuild exactly what every option would have sent.
                    limits = trajectory.limit_table(slice(at, at + 1))[0]
                    np.testing.assert_array_equal(limits, legal.limits)
                    np.testing.assert_array_equal(option_units(limits), legal.units)

                    # The stored choice must decode to exactly the move that was played.
                    if trajectory.source[at] == WAIT:
                        choice = None
                    else:
                        choice = SendChoice(int(trajectory.source[at]), int(trajectory.target[at]), int(trajectory.amount[at]))
                    played = moves.get(game.tick, {}).get(trajectory.seat)
                    if trajectory.log_probability is not None:
                        self.assertEqual(decode_action(choice, legal), played)
                    else:
                        self.assertEqual(choice, encode_action(played, legal))
                    # The units are those of the move itself, also where the option is only the nearest.
                    self.assertEqual(trajectory.units[at], WAIT if choice is None else played.units)
                    pointer[id(trajectory)] += 1
                    checked += 1
                game.step(moves.get(game.tick, {}))
            for trajectory in record.trajectories:
                self.assertEqual(pointer[id(trajectory)], len(trajectory))
        self.assertGreater(checked, 500)

    def test_positions_with_only_waiting_are_not_recorded(self):
        for record in self.self_play:
            for trajectory in record.trajectories:
                self.assertTrue(trajectory.allowed().any(axis=(1, 2, 3)).all())

    def test_model_seats_store_what_the_model_thought(self):
        for record in self.mixed:
            by_seat = {trajectory.seat: trajectory for trajectory in record.trajectories}
            net, teacher = by_seat[RED], by_seat[BLUE]
            self.assertEqual((net.player, teacher.player), ("net", "strategic_ai"))
            self.assertEqual(len(net.log_probability), len(net))
            self.assertEqual(len(net.value), len(net))
            self.assertIsNone(teacher.log_probability)
            self.assertIsNone(teacher.value)
            # The teacher's waits are decisions too; the network must learn when not to move.
            self.assertTrue((teacher.source == WAIT).any())

    def test_stored_probabilities_can_be_recomputed_from_the_trajectory(self):
        checked = 0
        for record in self.self_play[:2]:
            for trajectory in record.trajectories:
                decisions = slice(0, 200)
                with torch.no_grad():
                    output = self.model(trajectory_input(trajectory, decisions))
                route, amount = trajectory_choices(trajectory, decisions)
                recomputed = log_probability(output, route, amount)
                np.testing.assert_allclose(recomputed.numpy(), trajectory.log_probability[decisions], atol=1e-4)
                np.testing.assert_allclose(output.value.numpy(), trajectory.value[decisions], atol=1e-4)
                checked += len(route)
        self.assertGreater(checked, 300)

    def test_outcomes_agree_between_the_two_seats(self):
        pairs = {"win": "loss", "loss": "win", "draw": "draw"}
        for record in self.self_play + self.mixed:
            red, blue = sorted(record.trajectories, key=lambda trajectory: trajectory.seat)
            self.assertEqual((red.seat, blue.seat), (RED, BLUE))
            self.assertEqual(blue.outcome, pairs[red.outcome])
            expected = "draw" if record.winner is None else "win" if record.winner == RED else "loss"
            self.assertEqual(red.outcome, expected)

    def test_time_limit_ends_the_game_as_a_draw(self):
        records = play_games(matchups("net", "net", seeds=[0]), self.models, seed=1, max_seconds=2)
        self.assertEqual(records[0].ticks, 20)
        self.assertIsNone(records[0].winner)
        self.assertEqual({trajectory.outcome for trajectory in records[0].trajectories}, {"draw"})

    def test_the_same_seed_or_greedy_play_gives_the_same_games(self):
        def play(**settings):
            return play_games(matchups("net", "net", seeds=range(3)), self.models, max_seconds=15, **settings)

        first, second, other = play(seed=5), play(seed=5), play(seed=6)
        self.assertEqual([r.actions for r in first], [r.actions for r in second])
        self.assertTrue(same_trajectories(first, second))
        self.assertNotEqual([r.actions for r in first], [r.actions for r in other])
        # Greedy play needs no seed.
        self.assertEqual([r.actions for r in play(greedy=True)], [r.actions for r in play(greedy=True)])

    def test_decision_interval_spaces_out_model_decisions(self):
        records = play_games(
            matchups("net", "strategic_ai", seeds=[0, 1]),
            self.models,
            seed=1,
            max_seconds=20,
            decision_interval=3,
        )
        for record in records:
            for trajectory in record.trajectories:
                if trajectory.player == "net":
                    self.assertTrue((trajectory.tick % 3 == 0).all())
                else:
                    self.assertTrue((trajectory.tick % 3 != 0).any())
            self.assertTrue(all(tick % 3 == 0 for tick, seat, *_ in record.actions if seat == RED))

    def test_a_send_gap_spaces_out_that_seats_sends_only(self):
        played = [
            Matchup(generate_map(0).definition, Seat("net", True, min_send_gap=10), Seat("net", True)),
            Matchup(generate_map(1).definition, Seat("strategic_ai", True, min_send_gap=5), Seat("net", True)),
        ]
        records = play_games(played, self.models, seed=1, max_seconds=30)
        for record, gap in zip(records, (10, 5)):
            sends = {seat: [tick for tick, sender, *_ in record.actions if sender == seat] for seat in (RED, BLUE)}
            self.assertGreater(len(sends[RED]), 1)
            self.assertTrue((np.diff(sends[RED]) >= gap).all())
            self.assertTrue((np.diff(sends[BLUE]) < gap).any())  # the other seat is not limited
            # Ticks spent waiting out the gap are not decisions, so none is recorded.
            limited = next(trajectory for trajectory in record.trajectories if trajectory.seat == RED)
            last_send = [max([sent for sent in sends[RED] if sent < tick], default=-gap) for tick in limited.tick]
            self.assertTrue((limited.tick - np.array(last_send) >= gap).all())
            self.assertEqual(replay(record).winner(), record.winner)

    def test_unrecorded_seats_leave_no_trajectory(self):
        played = [Matchup(generate_map(0).definition, Seat("net", record=True), Seat("strategic_ai"))]
        record = play_games(played, self.models, seed=1, max_seconds=15)[0]
        self.assertEqual([trajectory.seat for trajectory in record.trajectories], [RED])
        self.assertEqual(count_decisions([record]), len(record.trajectories[0]))

    def test_worker_processes_play_the_same_games_as_this_one(self):
        scripted = matchups("strategic_ai", "aggressive_ai", seeds=range(5))
        from_pool = pool.play(scripted, max_seconds=40)
        here = play_games(scripted, max_seconds=40)
        self.assertEqual([r.matchup for r in from_pool], scripted)
        self.assertEqual([(r.winner, r.ticks, r.actions) for r in from_pool], [(r.winner, r.ticks, r.actions) for r in here])
        self.assertTrue(same_trajectories(from_pool, here))


class TemperatureTests(unittest.TestCase):
    """Networks drawing their moves at a temperature (see ``policy.choose``), here and through the server."""

    @classmethod
    def setUpClass(cls):
        cls.models = {"net": init_model("random", seed=0).model, "other": init_model("random", seed=1).model}

    def test_temperatures_must_be_above_zero(self):
        with self.assertRaises(ValueError):
            play_games(matchups("net", "net", seeds=[0]), self.models, max_seconds=1, temperatures={"net": 0.0})
        with self.assertRaises(ValueError):
            InferenceServer(self.models, "cpu", temperatures={"other": -1.0})

    def test_the_server_draws_by_the_sharpened_odds(self):
        record = play_games(matchups("net", "strategic_ai", seeds=[0]), self.models, seed=1, max_seconds=5)[0]
        trajectory = next(t for t in record.trajectories if t.player == "net")
        with torch.no_grad():
            output = self.models["net"].eval()(trajectory_input(trajectory, slice(0, 1)))
        # Clearly different odds for the usable routes, so sharpening shows.
        count = 4000
        logits = output.route_logits[0].clone()
        usable = logits > ILLEGAL
        logits[usable] = torch.linspace(0.0, 2.0, int(usable.sum()))
        many = replace(repeated(output, 0, count), route_logits=logits.expand(count, -1))

        server = InferenceServer(self.models, "cpu", temperatures={"net": 0.5})
        self.addCleanup(server.close)
        self.assertEqual(server.sharpen, [2.0, 1.0])
        route, _ = server._choose(many, torch.Generator().manual_seed(0), server.sharpen[0])
        expected = torch.softmax(logits * 2, dim=0)
        counts = torch.bincount(route, minlength=len(expected)).float() / count
        spread = (expected * (1 - expected) / count).sqrt()
        self.assertTrue(((counts - expected).abs() <= 5 * spread + 1e-3).all())


if __name__ == "__main__":
    unittest.main()
