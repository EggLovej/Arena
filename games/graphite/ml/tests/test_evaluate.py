import tempfile
import unittest
from pathlib import Path

from game import Game
from game_types import BLUE, NEUTRAL, RED, BaseDefinition, EdgeDefinition, MapDefinition, SendAction
from loading import load_map

from ml.evaluation.evaluate import (
    FIRST_EVALUATION_SEED,
    Evaluation,
    MatchPlan,
    MatchResult,
    evaluate,
    evaluation_maps,
    fresh_ai,
    keeps_memory,
    load_player,
    outcome_for,
    play_match,
)
from ml.models.checkpoint import init_model, save_checkpoint


def duel_map():
    return MapDefinition(
        name="duel",
        width=300,
        height=100,
        bases=(
            BaseDefinition("left", RED, 20, 0, 20, 50),
            BaseDefinition("right", BLUE, 5, 0, 280, 50),
        ),
        edges=(EdgeDefinition("left", "right", 0.2),),
    )


def wait(state):
    return None


def results_with(outcomes, opponent="aggressive_ai", map_set="generated"):
    return [
        MatchResult(MatchPlan(map_set, str(index), RED, opponent, 0), outcome, 60.0)
        for index, outcome in enumerate(outcomes)
    ]


class OutcomeTests(unittest.TestCase):
    def test_win_loss_and_draw(self):
        def attack(state):
            return SendAction("left", "right", 20) if state.tick == 0 else None

        won = play_match(duel_map(), attack, wait)
        self.assertEqual(outcome_for(won, RED), "win")
        self.assertEqual(outcome_for(won, BLUE), "loss")
        self.assertLess(won.time_seconds, 1.0)

        timed_out = play_match(duel_map(), wait, wait, max_seconds=2.0)
        self.assertEqual(outcome_for(timed_out, RED), "draw")
        self.assertEqual(outcome_for(timed_out, BLUE), "draw")
        self.assertAlmostEqual(timed_out.time_seconds, 2.0)

    def test_eliminated_player_has_lost_even_if_the_game_goes_on(self):
        game = Game(load_map("triangle"))
        for base in game.bases.values():
            if base.owner == RED:
                base.owner = NEUTRAL
        self.assertIsNone(game.winner())
        self.assertEqual(outcome_for(game, RED), "loss")
        self.assertEqual(outcome_for(game, BLUE), "draw")


class SummaryTests(unittest.TestCase):
    def test_score_counts_draws_as_half(self):
        summary = Evaluation("x", results_with(["win", "win", "draw", "loss"])).summary()
        self.assertEqual((summary.games, summary.wins, summary.draws, summary.losses), (4, 2, 1, 1))
        self.assertAlmostEqual(summary.score, 0.625)
        self.assertAlmostEqual(summary.decided, 0.75)

    def test_margin_shrinks_with_more_games_and_is_never_zero(self):
        few = Evaluation("x", results_with(["loss"] * 10)).summary()
        many = Evaluation("x", results_with(["loss"] * 1000)).summary()
        self.assertGreater(few.margin, many.margin)
        self.assertGreater(many.margin, 0)

    def test_summary_can_be_split_by_opponent_and_map_set(self):
        results = results_with(["win", "win"], "aggressive_ai", "bundled")
        results += results_with(["loss", "loss", "loss"], "strategic_ai", "generated")
        evaluation = Evaluation("x", results)
        self.assertEqual(evaluation.summary(opponent="aggressive_ai").score, 1.0)
        self.assertEqual(evaluation.summary(opponent="strategic_ai").score, 0.0)
        self.assertEqual(evaluation.summary(map_set="bundled").games, 2)
        self.assertEqual(evaluation.summary().games, 5)
        self.assertIn("vs strategic_ai", evaluation.table())
        self.assertIn("overall", evaluation.table())


class EvaluateTests(unittest.TestCase):
    def test_evaluation_maps_are_two_player_and_reserved(self):
        maps = evaluation_maps(generated_maps=5)
        bundled = [key for map_set, key in maps if map_set == "bundled"]
        generated = [int(key) for map_set, key in maps if map_set == "generated"]
        self.assertIn("classic", bundled)
        self.assertNotIn("triangle", bundled)
        self.assertFalse(any(name.startswith("generated_") for name in bundled))
        self.assertEqual(generated, list(range(FIRST_EVALUATION_SEED, FIRST_EVALUATION_SEED + 5)))
        self.assertEqual(evaluation_maps(3, bundled_maps=False), [("generated", str(FIRST_EVALUATION_SEED + i)) for i in range(3)])

    def test_fresh_maps_can_be_reached_by_skipping_the_usual_ones(self):
        usual = evaluation_maps(30, bundled_maps=False)
        fresh = evaluation_maps(5, bundled_maps=False, skip_generated=30)
        self.assertEqual(fresh, [("generated", str(FIRST_EVALUATION_SEED + 30 + i)) for i in range(5)])
        self.assertFalse(set(fresh) & set(usual))

    def test_players_load_by_name(self):
        state = Game(duel_map()).view_for(RED)
        for name in ("strategic_ai", "untrained", "untrained:3"):
            action = load_player(name, seed=0)(state)
            self.assertTrue(action is None or isinstance(action, SendAction))
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(Path(directory) / "model.pt", init_model("random", seed=1).model)
            self.assertTrue(callable(load_player(str(path), seed=0)))

    def test_an_ai_with_memory_gets_a_fresh_player_every_time(self):
        self.assertTrue(keeps_memory("atlas_clean_ai"))
        self.assertIsNot(fresh_ai("atlas_clean_ai"), fresh_ai("atlas_clean_ai"))
        self.assertIsNot(load_player("atlas_clean_ai"), load_player("atlas_clean_ai"))
        # Plain AIs are shared, and names that are not AI files have no memory to keep.
        self.assertFalse(keeps_memory("strategic_ai"))
        self.assertIs(fresh_ai("strategic_ai"), fresh_ai("strategic_ai"))
        self.assertFalse(keeps_memory("value:ml/runs/latest.pt"))

    def test_every_pairing_is_played_from_both_seats(self):
        seen = []
        evaluation = evaluate(
            "strategic_ai",
            opponents=["aggressive_ai", "careful_ai"],
            generated_maps=3,
            bundled_maps=False,
            workers=1,
            on_progress=lambda results, total: seen.append((len(results), total)),
        )
        self.assertEqual(len(evaluation.results), 2 * 3 * 2)
        self.assertEqual(seen, [(count, 12) for count in range(1, 13)])
        seats = {(r.plan.opponent, r.plan.map_key, r.plan.candidate_seat) for r in evaluation.results}
        self.assertEqual(len(seats), 12)
        # The strongest scripted AI should clearly beat the two simple ones.
        self.assertGreater(evaluation.summary().score, 0.7)

    def test_same_run_gives_the_same_results(self):
        settings = dict(opponents=["aggressive_ai"], generated_maps=2, bundled_maps=False, workers=1)
        first = evaluate("untrained", **settings)
        second = evaluate("untrained", **settings)
        self.assertEqual(first.results, second.results)

    def test_parallel_run_matches_a_single_worker(self):
        settings = dict(opponents=["aggressive_ai"], generated_maps=2, bundled_maps=False)
        single = evaluate("untrained", workers=1, **settings)
        parallel = evaluate("untrained", workers=2, **settings)
        self.assertEqual(single.results, parallel.results)


if __name__ == "__main__":
    unittest.main()
