import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import ml.training.runlog
from game_types import BLUE, RED
from ml.evaluation.evaluate import Evaluation, MatchPlan, MatchResult
from ml.models.checkpoint import init_model, load_checkpoint, save_checkpoint
from ml.play.rollout import GameRecord, Seat
from ml.training.league import LEARNER, League, LeagueSettings, fit_ratings, train_league
from ml.training.runlog import read_metrics
from ml.training.selfplay import SELF, START
from ml.training.train import parse_args


class FakePool:
    """Stands in for the rollout workers: the stronger player always wins, equals draw."""

    def __init__(self, strength):
        self.strength = strength
        self.calls = []

    def play(self, matchups, models=None, seed=None, **settings):
        self.calls.append((matchups, models or {}))
        records = []
        for matchup in matchups:
            red, blue = self.strength[matchup.red.player], self.strength[matchup.blue.player]
            winner = None if red == blue else RED if red > blue else BLUE
            records.append(GameRecord(matchup, winner, 600, []))
        return records


class FakeLog:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.name = "fake"

    def say(self, message):
        pass


def evaluation_of(scores, games=16):
    """An evaluation with this many games per opponent at the given scores."""

    results = []
    for opponent, score in scores.items():
        for game in range(games):
            outcome = "win" if game < round(score * games) else "loss"
            results.append(MatchResult(MatchPlan("generated", str(game), RED, opponent, 0), outcome, 60.0))
    return Evaluation("x", results)


class RatingTests(unittest.TestCase):
    def test_a_rating_gap_matches_the_score_it_predicts(self):
        ratings = fit_ratings({("a", "b"): [750.0, 1000]})
        # 75% corresponds to about 191 points.
        self.assertAlmostEqual(ratings["a"] - ratings["b"], 191, delta=5)

    def test_even_results_give_even_ratings(self):
        ratings = fit_ratings({("a", "b"): [50.0, 100]})
        self.assertAlmostEqual(ratings["a"], ratings["b"], places=3)

    def test_anchors_average_one_thousand(self):
        results = {("a", "b"): [70.0, 100], ("b", "c"): [60.0, 100], ("new", "a"): [80.0, 100]}
        ratings = fit_ratings(results, anchors=("a", "b", "c"))
        self.assertAlmostEqual((ratings["a"] + ratings["b"] + ratings["c"]) / 3, 1000, places=3)
        self.assertGreater(ratings["new"], ratings["a"])
        self.assertGreater(ratings["a"], ratings["b"])
        self.assertGreater(ratings["b"], ratings["c"])

    def test_a_perfect_record_still_gets_a_finite_rating(self):
        ratings = fit_ratings({("a", "b"): [20.0, 20]})
        self.assertTrue(np.isfinite(ratings["a"]))
        self.assertGreater(ratings["a"] - ratings["b"], 400)
        self.assertLess(ratings["a"] - ratings["b"], 2000)

    def test_all_matches_are_explained_together(self):
        # "c" never met "a", but beat "b", who drew with "a": so c should rate above a.
        ratings = fit_ratings({("a", "b"): [50.0, 100], ("c", "b"): [70.0, 100]})
        self.assertGreater(ratings["c"], ratings["a"])
        self.assertAlmostEqual(ratings["a"], ratings["b"], delta=15)


class LeagueTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        self.model = init_model("random", seed=0).model
        self.strength = {
            "aggressive_ai": 1, "careful_ai": 2, "strategic_ai": 3,
            "champion 0": 4, SELF: 5,
        }

    def league(self, **changes):
        defaults = dict(games_per_round=20, promotion_every=5, promotion_games=8, evaluation_maps=2)
        settings = LeagueSettings(**{**defaults, **changes})
        league = League(settings)
        pool = FakePool(self.strength)
        start_scores = {"aggressive_ai": 1.0, "careful_ai": 1.0, "strategic_ai": 1.0, START: 0.5}
        league.begin(self.model, evaluation_of(start_scores), pool, FakeLog(self.folder), np.random.default_rng(0))
        return league, pool

    def test_the_starting_model_is_the_first_champion(self):
        league, _ = self.league()
        self.assertEqual([champion.name for champion in league.champions], ["champion 0"])
        self.assertEqual(set(league.ratings), {"aggressive_ai", "careful_ai", "strategic_ai", "champion 0"})
        # The scripted AIs define the scale; the order follows who beats whom.
        scripted = [league.ratings[name] for name in ("aggressive_ai", "careful_ai", "strategic_ai")]
        self.assertAlmostEqual(sum(scripted) / 3, 1000, places=3)
        self.assertEqual(scripted, sorted(scripted))
        self.assertGreater(league.ratings["champion 0"], league.ratings["strategic_ai"])

    def test_games_are_shared_out_between_self_champions_and_scripted_ais(self):
        league, _ = self.league()
        matchups = league.matchups(np.random.default_rng(0))
        kinds = Counter()
        for matchup in matchups:
            players = {matchup.red.player, matchup.blue.player}
            if players == {SELF}:
                kinds["self"] += 1
                self.assertEqual(matchup.red, Seat(SELF, True))
                self.assertEqual(matchup.blue, Seat(SELF, True))
            else:
                other = (players - {SELF}).pop()
                kinds["champion" if other.startswith("champion") else "scripted"] += 1
                # Only the network's seat is recorded.
                self.assertEqual(sorted((matchup.red.record, matchup.blue.record)), [False, True])
        self.assertEqual(kinds, {"self": 10, "champion": 6, "scripted": 4})
        seats = [m.red.player == SELF for m in matchups if {m.red.player, m.blue.player} != {SELF}]
        self.assertIn(True, seats)
        self.assertIn(False, seats)
        self.assertEqual(set(league.models()), {"champion 0"})

    def test_opponents_it_struggles_against_are_met_more_often(self):
        league, _ = self.league(self_share=0.0, champion_share=0.0)
        league.recent = {"aggressive_ai": 0.95, "careful_ai": 0.95, "strategic_ai": 0.1}
        rng = np.random.default_rng(0)
        met = Counter()
        for _ in range(30):
            for matchup in league.matchups(rng):
                met[({matchup.red.player, matchup.blue.player} - {SELF}).pop()] += 1
        self.assertGreater(met["strategic_ai"], 5 * met["aggressive_ai"])
        self.assertGreater(met["aggressive_ai"], 0)

    def test_results_in_training_update_what_it_struggles_against(self):
        league, pool = self.league()
        records = pool.play(league.matchups(np.random.default_rng(0)), league.models())
        league.end_of_round(1, records, self.model, None, pool, np.random.default_rng(0))
        # The network is the strongest player here, so it won every game against the others.
        self.assertGreater(league.recent["champion 0"], 0.6)
        statistics = league.statistics(records)
        self.assertEqual(statistics["training wins/vs past champions"], 1.0)
        self.assertEqual(statistics["decided games/in self-play"], 0.0)

    def test_nothing_is_decided_between_promotion_rounds(self):
        league, pool = self.league()
        report = league.end_of_round(3, [], self.model, None, pool, np.random.default_rng(0))
        self.assertIs(report.promoted, False)
        self.assertEqual(report.metrics, {})
        self.assertEqual(len(league.champions), 1)

    def test_beating_the_champion_clearly_means_promotion(self):
        league, pool = self.league()
        before = dict(league.ratings)
        evaluation = evaluation_of({"aggressive_ai": 1.0, "careful_ai": 1.0, "strategic_ai": 1.0, START: 1.0})
        report = league.end_of_round(5, [], self.model, evaluation, pool, np.random.default_rng(0))

        self.assertIs(report.promoted, True)
        self.assertEqual([champion.name for champion in league.champions], ["champion 0", "champion 1"])
        self.assertEqual(league.champions[-1].round, 5)
        self.assertEqual(report.metrics["promotion match/score against the champion"], 1.0)
        self.assertEqual(report.metrics["promotion match/needed for promotion"], 0.55)
        self.assertIn("promoted to champion 1", report.summary)

        # The new champion is rated above the one it beat, and nothing is left under the temporary name.
        self.assertGreater(league.ratings["champion 1"], league.ratings["champion 0"])
        self.assertAlmostEqual(report.metrics["elo/champion"], league.ratings["champion 1"], places=3)
        self.assertAlmostEqual(report.metrics["elo/current network"], league.ratings["champion 1"], places=3)
        self.assertNotIn(LEARNER, league.ratings)
        self.assertFalse(any(LEARNER in pair for pair in league.results))
        self.assertGreater(league.ratings["champion 0"], before["strategic_ai"])

        # The promotion match was played from both seats against the reigning champion.
        matchups, models = pool.calls[-1]
        self.assertEqual(len(matchups), 8)
        self.assertEqual(sum(m.red.player == SELF for m in matchups), 4)
        self.assertEqual(set(models), {SELF, "champion 0"})
        # From now on the new champion is an opponent too.
        self.assertEqual(set(league.models()), {"champion 0", "champion 1"})

    def test_failing_the_match_changes_nothing(self):
        self.strength[SELF] = 4  # exactly as strong as the champion: every game is a draw
        # A long match, so the rating rests on the games and not on the starting assumption.
        league, pool = self.league(promotion_games=400)
        results_before = json.dumps(sorted(league.results.items()))
        report = league.end_of_round(5, [], self.model, None, pool, np.random.default_rng(0))

        self.assertIs(report.promoted, False)
        self.assertEqual(report.metrics["promotion match/score against the champion"], 0.5)
        self.assertIn("not promoted", report.summary)
        self.assertEqual(len(league.champions), 1)
        self.assertEqual(json.dumps(sorted(league.results.items())), results_before)
        # Drawing every game of a long match means the same rating as the champion.
        self.assertAlmostEqual(report.metrics["elo/current network"], report.metrics["elo/champion"], delta=5)
        self.assertNotIn(LEARNER, league.ratings)

    def test_only_the_newest_champions_stay_as_opponents(self):
        league, pool = self.league(max_champions=2)
        for round_number in (5, 10, 15):
            self.strength[f"champion {len(league.champions)}"] = 4
            league.end_of_round(round_number, [], self.model, None, pool, np.random.default_rng(0))
        self.assertEqual(len(league.champions), 4)
        self.assertEqual(set(league.models()), {"champion 2", "champion 3"})
        # Older champions keep their place in the ratings.
        self.assertIn("champion 0", league.ratings)

    def test_the_league_is_written_to_a_readable_file(self):
        league, pool = self.league()
        league.end_of_round(5, [], self.model, None, pool, np.random.default_rng(0))
        state = json.loads((self.folder / "league.json").read_text())
        self.assertEqual([c["name"] for c in state["champions"]], ["champion 0", "champion 1"])
        self.assertEqual(state["champions"][1]["file"], "round_0005.pt")
        self.assertEqual(state["promotion_matches"][0]["promoted"], True)
        self.assertEqual(state["promotion_matches"][0]["games"], 8)
        self.assertIn("strategic_ai", state["ratings"])

        table = league.closing_report()
        self.assertLess(table.index("champion 1"), table.index("champion 0"))
        self.assertLess(table.index("champion 0"), table.index("aggressive_ai"))


class LeagueResumeTests(LeagueTests):
    def played_league(self):
        """A league after one promotion and some training games, with its round files on disk."""

        league, pool = self.league()
        league._remember([GameRecord(matchup, BLUE, 600, []) for matchup in league.matchups(np.random.default_rng(1))])
        league.end_of_round(5, [], self.model, None, pool, np.random.default_rng(0))
        save_checkpoint(self.folder / "round_0000.pt", self.model)
        save_checkpoint(self.folder / "round_0005.pt", self.model)
        return league

    def saved_and_loaded(self, state):
        # The state travels inside a checkpoint, which only loads plain data.
        torch.save(state, self.folder / "state.pt")
        return torch.load(self.folder / "state.pt", weights_only=True)

    def test_a_restored_league_remembers_everything(self):
        league = self.played_league()
        restored = League(league.settings)
        restored.restore(self.saved_and_loaded(league.state()), FakeLog(self.folder))

        self.assertEqual([(c.name, c.round) for c in restored.champions], [("champion 0", 0), ("champion 1", 5)])
        self.assertEqual(restored.results, league.results)
        self.assertEqual(restored.recent, league.recent)
        self.assertEqual(restored.promotions, league.promotions)
        for name, rating in league.ratings.items():
            self.assertAlmostEqual(restored.ratings[name], rating, places=6)
        self.assertEqual(set(restored.models()), {"champion 0", "champion 1"})
        self.assertEqual(restored.state(), league.state())

    def test_a_deleted_champion_file_is_reported(self):
        league = self.played_league()
        (self.folder / "round_0005.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "champion 1"):
            League(league.settings).restore(league.state(), FakeLog(self.folder))

    def test_a_league_saved_without_its_state_cannot_be_resumed(self):
        with self.assertRaisesRegex(ValueError, "before league runs could be resumed"):
            League(LeagueSettings()).restore({}, FakeLog(self.folder))


class LeagueRunTests(unittest.TestCase):
    def test_a_small_league_run_leaves_a_complete_run_folder(self):
        settings = LeagueSettings(
            rounds=2,
            games_per_round=8,
            decisions_per_game=100,
            passes=2,
            batch_size=128,
            max_seconds=20,
            value_only_rounds=1,
            evaluation_maps=1,
            evaluate_every=1,
            final_evaluation=False,
            promotion_every=1,
            promotion_games=4,
            workers=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(ml.training.runlog, "RUNS_DIRECTORY", Path(directory)), mock.patch("builtins.print") as printed:
                best = train_league("tiny", settings, device="cpu")
                rounds = read_metrics("tiny")
            folder = Path(directory) / "tiny"
            names = sorted(entry.name for entry in folder.iterdir())
            state = json.loads((folder / "league.json").read_text())
            latest = load_checkpoint(folder / "latest.pt")
            self.assertTrue(best.exists())

        self.assertEqual(
            names,
            [
                "best.pt", "latest.pt", "league.json", "metrics.jsonl",
                "round_0000.pt", "round_0001.pt", "round_0002.pt", "settings.json",
            ],
        )
        self.assertEqual([entry["round"] for entry in rounds], [0, 1, 2])
        for entry in rounds[1:]:
            for name in (
                "promotion match/score against the champion", "promotion match/needed for promotion",
                "elo/current network", "elo/champion", "score/vs strategic_ai",
            ):
                self.assertIn(name, entry)
        self.assertEqual(len(state["promotion_matches"]), 2)
        self.assertEqual(state["champions"][0], {"name": "champion 0", "round": 0, "file": "round_0000.pt", "rating": state["champions"][0]["rating"]})
        self.assertEqual(latest.history[-1]["step"], "league")
        self.assertTrue(any("League table" in str(call.args[0]) for call in printed.call_args_list if call.args))

    def test_a_league_run_can_stop_and_carry_on_in_a_new_process(self):
        settings = LeagueSettings(
            rounds=3,
            games_per_round=6,
            decisions_per_game=60,
            passes=1,
            batch_size=64,
            max_seconds=15,
            value_only_rounds=0,
            evaluation_maps=0,
            final_evaluation=False,
            promotion_every=1,
            promotion_games=4,
            # Any score promotes, so every round adds a champion that must survive the restart.
            promotion_score=0.0,
            workers=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "split"
            with mock.patch.object(ml.training.runlog, "RUNS_DIRECTORY", Path(directory)), mock.patch("builtins.print"):
                train_league("split", settings, device="cpu", stop_after=2)
                paused = load_checkpoint(folder / "latest.pt").training_state["opponents"]
                train_league("split", settings, device="cpu", resume=True)
                rounds = read_metrics("split")
                final = load_checkpoint(folder / "latest.pt").training_state["opponents"]
                table = json.loads((folder / "league.json").read_text())

        self.assertEqual([entry["round"] for entry in rounds], [1, 2, 3])
        self.assertEqual([name for name, _ in paused["champions"]], ["champion 0", "champion 1", "champion 2"])
        self.assertEqual(final["champions"], [["champion 0", 0], ["champion 1", 1], ["champion 2", 2], ["champion 3", 3]])
        # The third promotion match was played against the champion promoted before the restart.
        self.assertEqual([match["against"] for match in final["promotions"]], ["champion 0", "champion 1", "champion 2"])
        self.assertEqual(final["promotions"][:2], paused["promotions"])
        self.assertEqual(len(table["promotion_matches"]), 3)

    def test_the_train_command_offers_the_league(self):
        with mock.patch.object(sys, "argv", ["train", "league", "--run", "x", "--promotion-games", "64", "--final-evaluation", "no"]):
            args = parse_args()
        self.assertEqual((args.trainer, args.promotion_games, args.final_evaluation), ("league", 64, False))
        self.assertEqual(args.self_share, LeagueSettings().self_share)
        with mock.patch.object(sys, "argv", ["train", "selfplay", "--run", "x", "--final-evaluation", "yes"]):
            self.assertIs(parse_args().final_evaluation, True)


if __name__ == "__main__":
    unittest.main()
