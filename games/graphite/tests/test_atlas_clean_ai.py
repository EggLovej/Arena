"""Atlas Clean: plays legally, keeps no copy of any opponent, and keeps matches apart."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from game import Game
from game_types import BLUE, RED
from loading import load_ai, load_map

from ais import atlas_clean_ai
from ais._atlas_sim import Simulation
from ais.atlas_clean_ai import WORK_PER_TICK, AtlasClean, new_player


ROOT = Path(__file__).resolve().parent.parent


def play(players, map_name, ticks):
    """Play a short match; returns every move made, by tick and seat."""

    game = Game(load_map(map_name))
    moves = []
    for _ in range(ticks):
        actions = {seat: player(game.view_for(seat)) for seat, player in players.items()}
        moves.append(actions)
        errors = game.step(actions)
        assert not errors, errors
    return moves


class NoCopiesTests(unittest.TestCase):
    def test_loading_it_loads_no_other_ai_and_no_network(self):
        loaded = subprocess.run(
            [sys.executable, "-c", "import sys, ais.atlas_clean_ai; print(' '.join(sorted(sys.modules)))"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.split()
        ais_modules = {name for name in loaded if name.startswith("ais")}
        self.assertEqual(ais_modules, {"ais", "ais._atlas_policy", "ais._atlas_sim", "ais.atlas_clean_ai"})
        self.assertFalse({"torch", "numpy"} & set(loaded))


class PlayTests(unittest.TestCase):
    def test_legal_repeatable_moves_against_a_scripted_ai(self):
        first = play({RED: new_player(), BLUE: load_ai("strategic_ai")}, "ring", 35)
        second = play({RED: new_player(), BLUE: load_ai("strategic_ai")}, "ring", 35)
        self.assertEqual(first, second)
        self.assertTrue(any(moves[RED] for moves in first))
        # Asked twice about one view, it gives the same answer.
        player = new_player()
        view = Game(load_map("classic")).view_for(RED)
        self.assertEqual(player(view), player(view))

    def test_the_shared_player_can_hold_both_seats(self):
        play({RED: atlas_clean_ai.choose_action, BLUE: atlas_clean_ai.choose_action}, "classic", 25)


class SpreadPlanningTests(unittest.TestCase):
    def steps_per_decision(self, player, map_name, ticks):
        """Simulation steps done inside each decision, and how many plans were finished."""

        counted, real_step = [0], Simulation.step

        def counting_step(simulation, actions, step_ticks):
            counted[0] += 1
            return real_step(simulation, actions, step_ticks)

        game, opponent, per_decision = Game(load_map(map_name)), load_ai("strategic_ai"), []
        with mock.patch.object(Simulation, "step", counting_step):
            for _ in range(ticks):
                counted[0] = 0
                action = player(game.view_for(RED))
                per_decision.append(counted[0])
                self.assertFalse(game.step({RED: action, BLUE: opponent(game.view_for(BLUE))}))
        (match,) = player._matches.values()
        return per_decision, match.plans_done

    def test_no_decision_does_more_than_its_share(self):
        per_decision, plans_done = self.steps_per_decision(new_player(), "classic", 120)
        self.assertLessEqual(max(per_decision), WORK_PER_TICK)
        # A whole plan is about 2,500 steps, so plans keep finishing every few game seconds.
        self.assertGreaterEqual(plans_done, 3)

    def test_planning_all_at_once_is_still_available(self):
        per_decision, plans_done = self.steps_per_decision(AtlasClean(work_per_tick=None), "classic", 30)
        self.assertGreater(max(per_decision), 1000)
        self.assertEqual(plans_done, 3)  # ticks 0, 10, and 20


class SideBySideTests(unittest.TestCase):
    def test_matches_played_side_by_side_play_as_if_alone(self):
        # Same map and seat, different opponents: a player keyed only by map and seat would mix them up.
        opponents = (load_ai("strategic_ai"), load_ai("aggressive_ai"))
        alone = [play({RED: new_player(), BLUE: opponent}, "classic", 40) for opponent in opponents]

        games = [Game(load_map("classic")) for _ in opponents]
        players = [new_player() for _ in opponents]
        together = [[], []]
        for _ in range(40):
            for number, (game, player, opponent) in enumerate(zip(games, players, opponents)):
                actions = {RED: player(game.view_for(RED)), BLUE: opponent(game.view_for(BLUE))}
                together[number].append(actions)
                self.assertFalse(game.step(actions))
        self.assertEqual(together, alone)


if __name__ == "__main__":
    unittest.main()
