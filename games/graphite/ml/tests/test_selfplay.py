import copy
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import ml.training.runlog
from game_types import RED
from ml.encoding.state_encoder import BASE_FEATURES, GLOBAL_FEATURES, PAIR_FEATURES, UNITS_REFERENCE
from ml.map_generator import generate_map
from ml.models.checkpoint import init_model, load_checkpoint, save_checkpoint
from ml.models import amounts
from ml.models.policy import _free_amounts_for, log_probability
from ml.play.rollout import WAIT, GameRecord, Matchup, Seat, Trajectory, play_games
from ml.tests.support import trajectory_choices, trajectory_input
from ml.training.runlog import read_metrics
from ml.training.selfplay import (
    SELF,
    START,
    SelfPlaySettings,
    adapted_learning_rate,
    behaviour,
    entropy_bonus,
    front_potential,
    hint_weight,
    judge_moves,
    judged_decisions,
    make_matchups,
    play_evaluation,
    train_selfplay,
    update,
)
from ml.training.train import parse_args


def fake_trajectory(ticks, values, outcome, units_share=0.5, production_share=0.5):
    """Just the parts of a trajectory that judging reads."""

    count = len(ticks)
    overall = np.zeros((count, len(GLOBAL_FEATURES)), dtype=np.float32)
    overall[:, GLOBAL_FEATURES.index("my_units_share")] = units_share
    overall[:, GLOBAL_FEATURES.index("my_production_share")] = production_share
    empty = np.zeros(count, dtype=np.int16)
    return Trajectory(
        seat=RED,
        player=SELF,
        outcome=outcome,
        base_ids=("a", "b"),
        pair_features=np.zeros((2, 2, 3), dtype=np.float32),
        base_features=np.zeros((count, 2, 14), dtype=np.float32),
        global_features=overall,
        routes=np.array([[0, 1], [1, 0]]),
        route_features=np.zeros((count, 2, 3), dtype=np.float32),
        allowed_bits=np.zeros((count, 5), dtype=np.uint8),
        source=empty,
        target=empty,
        amount=empty,
        tick=np.array(ticks, dtype=np.int32),
        log_probability=np.zeros(count, dtype=np.float32),
        value=np.array(values, dtype=np.float32),
    )


class JudgingTests(unittest.TestCase):
    settings = SelfPlaySettings(discount=0.9, lookahead=0.5)

    def test_value_target_is_the_result_discounted_by_distance_to_the_end(self):
        trajectory = fake_trajectory([0, 3, 8], [0, 0, 0], "win")
        _, target = judge_moves(trajectory, final_tick=10, settings=self.settings)
        np.testing.assert_allclose(target, [0.9**10, 0.9**7, 0.9**2], rtol=1e-6)

        _, lost = judge_moves(replace(trajectory, outcome="loss"), 10, self.settings)
        np.testing.assert_allclose(lost, -target, rtol=1e-6)
        _, drawn = judge_moves(replace(trajectory, outcome="draw"), 10, self.settings)
        np.testing.assert_array_equal(drawn, [0, 0, 0])

    def test_full_lookahead_compares_the_result_with_what_was_expected(self):
        settings = replace(self.settings, lookahead=1.0)
        trajectory = fake_trajectory([0, 3, 8], [0.2, -0.1, 0.4], "win")
        advantage, target = judge_moves(trajectory, 10, settings)
        np.testing.assert_allclose(advantage, target - trajectory.value, rtol=1e-5)

    def test_no_lookahead_compares_only_with_the_next_decision(self):
        settings = replace(self.settings, lookahead=0.0)
        trajectory = fake_trajectory([0, 3, 8], [0.2, -0.1, 0.4], "win")
        advantage, target = judge_moves(trajectory, 10, settings)
        expected = [0.9**3 * -0.1 - 0.2, 0.9**5 * 0.4 - -0.1, target[-1] - 0.4]
        np.testing.assert_allclose(advantage, expected, rtol=1e-5)

    def test_a_correct_value_output_leaves_nothing_to_credit(self):
        ticks = [0, 4, 9]
        perfect = [0.9 ** (10 - tick) for tick in ticks]
        advantage, _ = judge_moves(fake_trajectory(ticks, perfect, "win"), 10, self.settings)
        np.testing.assert_allclose(advantage, [0, 0, 0], atol=1e-6)

    def test_a_single_decision_is_judged_by_the_result_alone(self):
        advantage, target = judge_moves(fake_trajectory([5], [0.3], "loss"), 10, self.settings)
        np.testing.assert_allclose(advantage, target - 0.3, rtol=1e-6)

    def test_the_rule_of_thumb_rewards_gaining_material(self):
        settings = replace(self.settings, lookahead=0.0)
        # Value output knows nothing (all zeros); material share climbs, then the game is drawn.
        climbing = fake_trajectory([0, 5, 9], [0, 0, 0], "draw")
        climbing.global_features[:, GLOBAL_FEATURES.index("my_production_share")] = [0.5, 0.7, 0.9]

        without, _ = judge_moves(climbing, 10, settings, hint=0.0)
        with_hint, _ = judge_moves(climbing, 10, settings, hint=1.0)
        np.testing.assert_allclose(without, [0, 0, 0], atol=1e-7)
        # The first two moves were followed by a better material position.
        self.assertGreater(with_hint[0], 0)
        self.assertGreater(with_hint[1], 0)

    def test_the_entropy_bonus_halves_towards_its_floor_and_never_below(self):
        settings = SelfPlaySettings(entropy_weight=0.001, entropy_half_life=100, entropy_floor=0.0002)
        self.assertAlmostEqual(entropy_bonus(settings, 1), 0.001)
        self.assertAlmostEqual(entropy_bonus(settings, 101), 0.0002 + 0.0008 / 2)
        self.assertAlmostEqual(entropy_bonus(settings, 201), 0.0002 + 0.0008 / 4)
        self.assertGreater(entropy_bonus(settings, 100_000), 0.0002 - 1e-12)
        # Without a half-life it stays where it started.
        self.assertEqual(entropy_bonus(SelfPlaySettings(entropy_weight=0.003), 500), 0.003)

    def test_the_rule_of_thumb_fades_out(self):
        settings = SelfPlaySettings(material_hint=0.5, material_hint_rounds=10)
        self.assertEqual(hint_weight(settings, 1), 0.5)
        self.assertAlmostEqual(hint_weight(settings, 6), 0.25)
        self.assertEqual(hint_weight(settings, 11), 0.0)
        self.assertEqual(hint_weight(settings, 500), 0.0)
        self.assertEqual(hint_weight(replace(settings, material_hint_rounds=0), 1), 0.0)


def line_trajectory(units, incoming):
    """Three bases in a row, A - B - C: A and B mine, C the enemy's. Travel A-B 2 s, B-C 1 s.

    ``units`` and ``incoming`` hold, per decision, my units at and on their way to each base.
    """

    count = len(units)
    features = np.zeros((count, 3, len(BASE_FEATURES)), dtype=np.float32)
    features[:, :2, BASE_FEATURES.index("is_mine")] = 1
    features[:, 2, BASE_FEATURES.index("is_enemy")] = 1
    scale = 1 / np.log1p(UNITS_REFERENCE)
    features[..., BASE_FEATURES.index("units")] = np.log1p(np.array(units, dtype=np.float32)) * scale
    features[..., BASE_FEATURES.index("incoming_mine")] = np.log1p(np.array(incoming, dtype=np.float32)) * scale
    pairs = np.zeros((3, 3, len(PAIR_FEATURES)), dtype=np.float32)
    pairs[..., PAIR_FEATURES.index("shortest_path_time")] = [[0, 2, 3], [2, 0, 1], [3, 1, 0]]
    trajectory = fake_trajectory(list(range(count)), [0.0] * count, "draw")
    return replace(trajectory, base_ids=("a", "b", "c"), base_features=features, pair_features=pairs)


class PotentialTests(unittest.TestCase):
    def test_units_count_by_their_distance_to_the_nearest_enemy_base(self):
        trajectory = line_trajectory(
            # 10 at A (3 s from C) and 10 at B (1 s); then A's 10 on their way to B; then attacking C.
            units=[[10, 10, 0], [0, 10, 0], [0, 0, 0]],
            incoming=[[0, 0, 0], [0, 10, 0], [0, 0, 20]],
        )
        # The longest shortest path is 3 s: (10 x 3 + 10 x 1) / (20 x 3), then (20 x 1) / (20 x 3), then 0.
        np.testing.assert_allclose(front_potential(trajectory), [-40 / 60, -20 / 60, 0.0], atol=1e-4)

    def test_moves_are_credited_with_the_change_in_production_share(self):
        settings = SelfPlaySettings(discount=1.0, lookahead=0.0, production_potential=0.5)
        trajectory = fake_trajectory([0, 5, 9], [0, 0, 0], "draw")
        trajectory.global_features[:, GLOBAL_FEATURES.index("my_production_share")] = [0.4, 0.6, 0.6]
        advantage, _ = judge_moves(trajectory, 10, settings)
        # From 0.4 to 0.6 is +0.2 x 2 x 0.5; holding it earns nothing; at the end the potential is given back.
        np.testing.assert_allclose(advantage, [0.2, 0.0, -0.5 * (2 * 0.6 - 1)], atol=1e-6)

    def test_the_front_potential_credits_sending_units_forward(self):
        settings = SelfPlaySettings(discount=1.0, lookahead=0.0, front_potential=1.0)
        trajectory = line_trajectory(units=[[10, 10, 0], [0, 10, 0], [0, 10, 0]], incoming=[[0, 0, 0], [0, 10, 0], [0, 10, 0]])
        advantage, _ = judge_moves(trajectory, 10, settings)
        self.assertAlmostEqual(advantage[0], -20 / 60 - -40 / 60, places=4)
        self.assertAlmostEqual(advantage[1], 0.0, places=6)

    def test_both_are_off_by_default(self):
        trajectory = fake_trajectory([0, 5, 9], [0.1, 0.2, 0.3], "win")
        trajectory.global_features[:, GLOBAL_FEATURES.index("my_production_share")] = [0.1, 0.9, 0.5]
        plain, _ = judge_moves(trajectory, 10, SelfPlaySettings())
        off, _ = judge_moves(trajectory, 10, SelfPlaySettings(production_potential=0.0, front_potential=0.0))
        np.testing.assert_array_equal(plain, off)


class ValueErrorTests(unittest.TestCase):
    def records(self, values, outcome):
        trajectory = fake_trajectory([0, 5, 10], values, outcome)
        matchup = Matchup(generate_map(0).definition, Seat(SELF, True), Seat(SELF, True))
        from ml.play.rollout import GameRecord

        return [GameRecord(matchup, RED, 10, [], [trajectory])]

    def errors(self, values, outcome):
        settings = SelfPlaySettings(discount=1.0)
        with mock.patch("ml.training.selfplay.DecisionSet"):
            _, errors = judged_decisions(self.records(values, outcome), settings, 0.0, np.random.default_rng(0))
        return errors

    def test_saying_zero_costs_nothing_in_a_draw_and_everything_in_a_decided_game(self):
        self.assertEqual(self.errors([0, 0, 0], "draw")["value error/always saying 0"], 0.0)
        self.assertEqual(self.errors([0, 0, 0], "win")["value error/always saying 0"], 1.0)
        self.assertEqual(self.errors([0, 0, 0], "win")["value error/of the network"], 1.0)

    def test_a_network_that_sees_the_result_coming_beats_the_reference(self):
        errors = self.errors([0.8, 0.9, 1.0], "win")
        self.assertLess(errors["value error/of the network"], 0.05)
        self.assertEqual(errors["value error/always saying 0"], 1.0)

    def test_a_confidently_wrong_network_is_worse_than_the_reference(self):
        errors = self.errors([-0.9, -0.9, -0.9], "win")
        self.assertGreater(errors["value error/of the network"], errors["value error/always saying 0"])


class MatchupTests(unittest.TestCase):
    def test_self_play_records_both_seats(self):
        matchups = make_matchups(SelfPlaySettings(games_per_round=6), np.random.default_rng(0))
        self.assertEqual(len(matchups), 6)
        for matchup in matchups:
            self.assertEqual((matchup.red, matchup.blue), (Seat(SELF, True), Seat(SELF, True)))

    def test_scripted_opponents_are_met_from_both_seats_and_not_recorded(self):
        settings = SelfPlaySettings(games_per_round=8, opponents=(SELF, "strategic_ai"))
        matchups = make_matchups(settings, np.random.default_rng(0))
        against = [m for m in matchups if Seat("strategic_ai") in (m.red, m.blue)]
        self.assertEqual(len(against), 4)
        self.assertEqual(sum(m.red == Seat(SELF, True) for m in against), 2)
        self.assertEqual(sum(m.blue == Seat(SELF, True) for m in against), 2)

    def test_a_send_gap_limits_the_network_seats_only(self):
        settings = SelfPlaySettings(games_per_round=4, opponents=(SELF, "strategic_ai"), min_send_gap=10)
        for matchup in make_matchups(settings, np.random.default_rng(0)):
            for seat in (matchup.red, matchup.blue):
                self.assertEqual(seat.min_send_gap, 10 if seat.player == SELF else 0)

    def test_the_evaluation_limits_the_networks_of_the_run_only(self):
        class Pool:
            def play(self, matchups, models, seed):
                self.matchups = matchups
                return [GameRecord(matchup, None, 10, []) for matchup in matchups]

        pool = Pool()
        play_evaluation(pool, None, None, SelfPlaySettings(evaluation_maps=1, min_send_gap=10))
        self.assertIn(START, {matchup.blue.player for matchup in pool.matchups})
        for matchup in pool.matchups:
            for seat in (matchup.red, matchup.blue):
                self.assertEqual(seat.min_send_gap, 10 if seat.player in (SELF, START) else 0)

    def test_training_maps_differ_from_round_to_round(self):
        rng = np.random.default_rng(0)
        first = make_matchups(SelfPlaySettings(games_per_round=4), rng)
        second = make_matchups(SelfPlaySettings(games_per_round=4), rng)
        self.assertNotEqual([m.map.name for m in first], [m.map.name for m in second])


class UpdateTests(unittest.TestCase):
    def test_moves_judged_good_become_more_likely(self):
        model = init_model("random", seed=0).model
        settings = SelfPlaySettings(passes=3, batch_size=256, max_policy_change=1.0, entropy_weight=0.0, max_seconds=25)
        matchups = [Matchup(generate_map(seed).definition, Seat(SELF, True), Seat(SELF, True)) for seed in range(4)]
        records = play_games(matchups, {SELF: model}, seed=0, max_seconds=settings.max_seconds)

        # Pretend that waiting was always the good move and sending the bad one.
        for record in records:
            for trajectory in record.trajectories:
                trajectory.value[:] = 0.0
        rng = np.random.default_rng(0)
        decisions, _ = judged_decisions(records, settings, hint=0.0, rng=rng)
        for group in decisions._groups.values():
            data = group.join()
            data["target:advantage"] = np.where(data["route"] == 0, 1.0, -1.0).astype(np.float32)

        def chance_of_waiting():
            total = count = 0.0
            with torch.no_grad():
                for record in records:
                    for trajectory in record.trajectories:
                        output = model(trajectory_input(trajectory))
                        total += torch.softmax(output.route_logits, dim=1)[:, 0].sum().item()
                        count += len(trajectory)
            return total / count

        before = chance_of_waiting()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        learned = update(model, optimizer, decisions, settings, rng, "cpu")
        self.assertGreater(chance_of_waiting(), before + 0.02)
        self.assertGreater(learned["policy change/per round"], 0)
        self.assertEqual(learned["decisions learned from"], len(decisions))

    def test_free_amounts_judged_good_become_more_likely(self):
        model = init_model("random", seed=0, amounts="free", feature_set="v2").model
        settings = SelfPlaySettings(passes=3, batch_size=256, max_policy_change=1.0, entropy_weight=0.0, max_seconds=25)
        matchups = [Matchup(generate_map(seed).definition, Seat(SELF, True), Seat(SELF, True)) for seed in range(4)]
        records = play_games(matchups, {SELF: model}, seed=0, max_seconds=settings.max_seconds)
        rng = np.random.default_rng(0)
        decisions, _ = judged_decisions(records, settings, hint=0.0, rng=rng, free_amounts=True)
        # Pretend that sending exactly one unit was always good and any other amount bad.
        for group in decisions._groups.values():
            data = group.join()
            data["target:advantage"] = np.where(data["route"] == 0, 0.0, np.where(data["amount"] == 1, 1.0, -1.0)).astype(np.float32)

        def chance_of_one_unit():
            chances = []
            with torch.no_grad():
                for record in records:
                    for trajectory in record.trajectories:
                        route, _ = trajectory_choices(trajectory)
                        sent = route > 0
                        curves, most, needed = _free_amounts_for(model(trajectory_input(trajectory)), route)
                        one = amounts.log_probability(curves, most, needed, torch.ones_like(route)).exp()
                        chances += one[sent & (most > 1)].tolist()
            return float(np.mean(chances))

        before = chance_of_one_unit()
        learned = update(model, torch.optim.Adam(model.parameters(), lr=1e-3), decisions, settings, rng, "cpu")
        self.assertGreater(chance_of_one_unit(), before + 0.05)
        self.assertTrue(np.isfinite(learned["entropy/of the policy"]))
        # The entropy is logged in its two parts as well, which add up to it.
        self.assertGreater(learned["entropy/of the amounts"], 0)
        self.assertAlmostEqual(
            learned["entropy/of the route choice"] + learned["entropy/of the amounts"], learned["entropy/of the policy"], places=5
        )

    def test_a_confident_policy_is_held_in_place_while_only_the_value_learns(self):
        model = init_model("random", seed=0).model
        # Make the untrained network confident: the same preferences, fifty times stronger.
        with torch.no_grad():
            model.route_out.weight *= 50
        matchups = [Matchup(generate_map(seed).definition, Seat(SELF, True), Seat(SELF, True)) for seed in range(4)]
        records = play_games(matchups, {SELF: model}, seed=0, max_seconds=25)

        def shift_after_value_only_round(hold_weight):
            settings = SelfPlaySettings(passes=3, batch_size=256, max_seconds=25, hold_weight=hold_weight)
            rng = np.random.default_rng(0)
            decisions, _ = judged_decisions(records, settings, hint=0.0, rng=rng)
            for group in decisions._groups.values():
                # A value target far from what the network says, so the value has a lot to learn.
                group.join()["target:value_target"][:] = 1.0
            trained = copy.deepcopy(model)
            optimizer = torch.optim.Adam(trained.parameters(), lr=3e-4)
            update(trained, optimizer, decisions, settings, rng, "cpu", learn_moves=False)
            total = count = 0.0
            with torch.no_grad():
                for batch in decisions.batches(512):
                    new = log_probability(trained(batch.inputs), batch.route, batch.amount)
                    total += (batch.targets["log_probability"] - new).abs().sum().item()
                    count += len(batch)
            return total / count

        free = shift_after_value_only_round(hold_weight=0.0)
        held = shift_after_value_only_round(hold_weight=1.0)
        self.assertGreater(free, 0.05)
        self.assertLess(held, free / 3)

    def test_learning_stops_once_the_policy_has_moved_far_enough(self):
        model = init_model("random", seed=0).model
        with torch.no_grad():
            model.route_out.weight *= 50
        matchups = [Matchup(generate_map(seed).definition, Seat(SELF, True), Seat(SELF, True)) for seed in range(4)]
        records = play_games(matchups, {SELF: model}, seed=0, max_seconds=25)
        settings = SelfPlaySettings(passes=20, batch_size=128, max_seconds=25, max_policy_change=0.02)
        rng = np.random.default_rng(0)
        decisions, _ = judged_decisions(records, settings, hint=0.5, rng=rng)

        steps = []
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        original_step = optimizer.step
        optimizer.step = lambda *args, **kwargs: (steps.append(1), original_step(*args, **kwargs))[1]
        learned = update(model, optimizer, decisions, settings, rng, "cpu")
        full_run = settings.passes * -(-len(decisions) // settings.batch_size)
        self.assertLess(len(steps), full_run)
        self.assertGreater(learned["policy change/per round"], settings.max_policy_change)

    def test_step_size_follows_how_far_the_policy_moved(self):
        settings = SelfPlaySettings(target_policy_change=0.015, min_learning_rate=1e-5, max_learning_rate=1e-3)
        self.assertAlmostEqual(adapted_learning_rate(1e-4, 0.1, settings), 1e-4 / 1.5)
        self.assertAlmostEqual(adapted_learning_rate(1e-4, 0.001, settings), 1.5e-4)
        self.assertEqual(adapted_learning_rate(1e-4, 0.015, settings), 1e-4)
        self.assertEqual(adapted_learning_rate(1e-5, 0.1, settings), 1e-5)
        self.assertEqual(adapted_learning_rate(1e-3, 0.0, settings), 1e-3)

    def test_value_only_rounds_leave_the_moves_almost_alone(self):
        model = init_model("random", seed=0).model
        settings = SelfPlaySettings(passes=2, batch_size=256, max_seconds=20)
        matchups = [Matchup(generate_map(seed).definition, Seat(SELF, True), Seat(SELF, True)) for seed in range(3)]
        records = play_games(matchups, {SELF: model}, seed=0, max_seconds=settings.max_seconds)
        rng = np.random.default_rng(0)
        decisions, _ = judged_decisions(records, settings, hint=0.5, rng=rng)

        trajectory = records[0].trajectories[0]
        route, amount = trajectory_choices(trajectory)
        with torch.no_grad():
            before = log_probability(model(trajectory_input(trajectory)), route, amount)
            values_before = model(trajectory_input(trajectory)).value.clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        update(model, optimizer, decisions, settings, rng, "cpu", learn_moves=False)
        with torch.no_grad():
            after = log_probability(model(trajectory_input(trajectory)), route, amount)
            values_after = model(trajectory_input(trajectory)).value
        self.assertLess((after - before).abs().mean().item(), 0.05)
        self.assertGreater((values_after - values_before).abs().max().item(), 1e-4)


class CaptureLearningTests(unittest.TestCase):
    def test_a_network_with_a_capture_output_learns_where_its_sends_ended_up(self):
        torch.manual_seed(0)
        model = init_model("random", seed=0, capture_head=True).model
        settings = SelfPlaySettings(passes=4, batch_size=256, max_policy_change=1.0, capture_weight=1.0)
        matchups = [Matchup(generate_map(seed).definition, Seat(SELF, True), Seat(SELF, True)) for seed in range(4)]
        records = play_games(matchups, {SELF: model}, seed=0, max_seconds=40)
        rng = np.random.default_rng(0)
        decisions, _ = judged_decisions(records, settings, hint=0.0, rng=rng)
        self.assertGreater(np.isfinite(decisions.target("captured")).sum(), 50)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        first = update(model, optimizer, decisions, settings, rng, "cpu", learn_moves=False)
        for _ in range(3):
            last = update(model, optimizer, decisions, settings, rng, "cpu", learn_moves=False)
        self.assertLess(last["capture error/of the network"], first["capture error/of the network"])

    def test_a_network_without_one_reports_nothing(self):
        model = init_model("random", seed=0).model
        settings = SelfPlaySettings(passes=1, batch_size=256)
        records = play_games([Matchup(generate_map(1).definition, Seat(SELF, True), Seat(SELF, True))], {SELF: model}, seed=0, max_seconds=20)
        rng = np.random.default_rng(0)
        decisions, _ = judged_decisions(records, settings, hint=0.0, rng=rng)
        learned = update(model, torch.optim.Adam(model.parameters()), decisions, settings, rng, "cpu")
        self.assertNotIn("capture error/of the network", learned)


class BrokenNumbersTests(unittest.TestCase):
    """A batch with non-finite numbers is skipped; broken weights at the end are undone."""

    def setUp(self):
        self.model = init_model("random", seed=0).model
        self.settings = SelfPlaySettings(passes=1, batch_size=128, max_policy_change=1.0)
        records = play_games([Matchup(generate_map(1).definition, Seat(SELF, True), Seat(SELF, True))], {SELF: self.model}, seed=0, max_seconds=20)
        self.rng = np.random.default_rng(0)
        self.decisions, _ = judged_decisions(records, self.settings, hint=0.0, rng=self.rng)

    def test_a_batch_with_a_non_finite_advantage_is_skipped(self):
        for group in self.decisions._groups.values():
            data = group.join()
            data["target:advantage"][:] = np.nan
        before = [p.detach().clone() for p in self.model.parameters()]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        learned = update(self.model, optimizer, self.decisions, self.settings, self.rng, "cpu")
        self.assertGreater(learned["learning problems/batches skipped"], 0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, self.model.parameters())))

    def test_weights_broken_by_a_round_are_put_back(self):
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        original_step = optimizer.step

        def breaking_step(*args, **kwargs):
            original_step(*args, **kwargs)
            with torch.no_grad():
                next(self.model.parameters()).fill_(float("inf"))

        optimizer.step = breaking_step
        learned = update(self.model, optimizer, self.decisions, self.settings, self.rng, "cpu")
        self.assertEqual(learned["learning problems/round undone"], 1.0)
        for name, p in self.model.named_parameters():
            self.assertTrue(torch.equal(p, before[name]), name)

    def test_a_healthy_round_reports_no_problems(self):
        learned = update(self.model, torch.optim.Adam(self.model.parameters(), lr=1e-4), self.decisions, self.settings, self.rng, "cpu")
        self.assertNotIn("learning problems/batches skipped", learned)


class BehaviourTests(unittest.TestCase):
    def test_where_sends_and_units_go_is_measured(self):
        model = init_model("random", seed=0).model
        records = play_games([Matchup(generate_map(seed).definition, Seat(SELF, True), Seat(SELF, True)) for seed in range(2)], {SELF: model}, seed=0, max_seconds=30)
        metrics = behaviour(records)
        self.assertTrue(0 <= metrics["behaviour/sends to own bases"] <= 1)
        self.assertTrue(0 <= metrics["behaviour/units away from the front"] <= 1)
        sends = sum(int((t.source != WAIT).sum()) for r in records for t in r.trajectories)
        self.assertAlmostEqual(metrics["sends/per game and seat"], sends / 4)
        decisions = sum(len(t) for r in records for t in r.trajectories)
        self.assertAlmostEqual(metrics["behaviour/waits when it may send"], 1 - sends / decisions)


class SelfPlayRunTests(unittest.TestCase):
    def test_a_small_run_leaves_a_complete_run_folder(self):
        settings = SelfPlaySettings(
            rounds=2,
            games_per_round=6,
            decisions_per_game=100,
            passes=2,
            batch_size=128,
            max_seconds=25,
            value_only_rounds=1,
            evaluation_maps=1,
            evaluate_every=2,
            final_evaluation=False,
            workers=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(ml.training.runlog, "RUNS_DIRECTORY", Path(directory)), mock.patch("builtins.print"):
                best = train_selfplay("tiny", settings, device="cpu")
                rounds = read_metrics("tiny")
            names = sorted(entry.name for entry in (Path(directory) / "tiny").iterdir())
            latest = load_checkpoint(Path(directory) / "tiny" / "latest.pt")
            start = load_checkpoint(Path(directory) / "tiny" / "round_0000.pt")
            self.assertTrue(best.exists())

        self.assertEqual(
            names,
            ["best.pt", "latest.pt", "metrics.jsonl", "round_0000.pt", "round_0001.pt", "round_0002.pt", "settings.json"],
        )
        self.assertEqual([entry["round"] for entry in rounds], [0, 1, 2])
        for name in (
            "decided games/in self-play", "entropy/of the policy", "policy change/per round",
            "value error/of the network", "value error/always saying 0",
        ):
            self.assertIn(name, rounds[1])
        # Matches against the scripted AIs and the starting model: before training, then every second round.
        self.assertNotIn("score/vs strategic_ai", rounds[1])
        for entry in (rounds[0], rounds[2]):
            for opponent in ("aggressive_ai", "careful_ai", "strategic_ai", "starting model"):
                self.assertIn(f"score/vs {opponent}", entry)
        self.assertNotIn("entropy/of the policy", rounds[0])
        self.assertEqual(list(rounds[2])[2].split("/")[0], "score")

        self.assertEqual([entry["step"] for entry in start.history], ["random start"])
        self.assertEqual([entry["step"] for entry in latest.history], ["random start", "self-play"])
        self.assertEqual(latest.history[-1]["rounds"], 2)
        self.assertIn("optimizer", latest.training_state)
        changed = any(
            not torch.equal(a, b)
            for a, b in zip(start.model.state_dict().values(), latest.model.state_dict().values())
        )
        self.assertTrue(changed)

    def test_a_network_with_free_amounts_trains_with_the_server(self):
        settings = SelfPlaySettings(
            rounds=2, games_per_round=6, decisions_per_game=100, passes=2, batch_size=128, max_seconds=25,
            value_only_rounds=1, evaluation_maps=1, evaluate_every=2, final_evaluation=False, workers=2,
            inference="server",
        )
        with tempfile.TemporaryDirectory() as directory:
            start = Path(directory) / "free.pt"
            save_checkpoint(start, init_model("random", seed=0, amounts="free", feature_set="v2").model)
            with mock.patch.object(ml.training.runlog, "RUNS_DIRECTORY", Path(directory)), mock.patch("builtins.print"):
                train_selfplay("free", settings, init=str(start), device="cpu")
                rounds = read_metrics("free")
            latest = load_checkpoint(Path(directory) / "free" / "latest.pt")
        self.assertTrue(latest.model.free_amounts)
        self.assertGreater(rounds[2]["policy change/per round"], 0)
        self.assertIn("score/vs strategic_ai", rounds[2])
        self.assertTrue(all(np.isfinite(value) for value in rounds[2].values() if isinstance(value, float)))

    def test_a_run_can_stop_and_carry_on_in_a_new_process(self):
        settings = SelfPlaySettings(
            rounds=3,
            games_per_round=4,
            decisions_per_game=60,
            passes=1,
            batch_size=64,
            max_seconds=15,
            value_only_rounds=0,
            evaluation_maps=0,
            final_evaluation=False,
            workers=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "split"
            with mock.patch.object(ml.training.runlog, "RUNS_DIRECTORY", Path(directory)), mock.patch("builtins.print"):
                train_selfplay("split", settings, device="cpu", stop_after=2)
                first = read_metrics("split")
                paused = load_checkpoint(folder / "latest.pt")
                self.assertFalse((folder / "round_0003.pt").exists())

                train_selfplay("split", settings, device="cpu", resume=True)
                rounds = read_metrics("split")
                latest = load_checkpoint(folder / "latest.pt")

        self.assertEqual([entry["round"] for entry in first], [1, 2])
        self.assertEqual([entry["round"] for entry in rounds], [1, 2, 3])
        # Time keeps counting from where the first process stopped.
        self.assertGreaterEqual(rounds[2]["seconds"], rounds[1]["seconds"])

        self.assertEqual(paused.training_state["round"], 2)
        self.assertEqual(latest.training_state["round"], 3)
        # The optimizer carried on counting its steps instead of starting over.
        paused_steps = max(entry["step"] for entry in paused.training_state["optimizer"]["state"].values())
        final_steps = max(entry["step"] for entry in latest.training_state["optimizer"]["state"].values())
        self.assertGreater(final_steps, paused_steps)
        # One history entry for this run, with the final round count.
        steps = [(entry["step"], entry.get("rounds")) for entry in latest.history]
        self.assertEqual(steps, [("random start", None), ("self-play", 3)])

    def test_the_train_command_offers_self_play(self):
        with mock.patch.object(sys, "argv", ["train", "selfplay", "--run", "x", "--rounds", "7", "--opponents", "self", "careful_ai"]):
            args = parse_args()
        self.assertEqual((args.trainer, args.rounds, args.opponents), ("selfplay", 7, ["self", "careful_ai"]))
        self.assertEqual(args.learning_rate, SelfPlaySettings().learning_rate)


if __name__ == "__main__":
    unittest.main()
