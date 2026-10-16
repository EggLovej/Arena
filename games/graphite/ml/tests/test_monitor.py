import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import ml.training.monitor
import ml.training.runlog
from ml.training.monitor import Monitor, format_value, group_metrics, round_ticks, value_axis
from ml.training.runlog import RunLog


class ChartMathTests(unittest.TestCase):
    def test_metrics_are_grouped_by_the_part_before_the_slash(self):
        rounds = [
            {"round": 1, "seconds": 3.0, "score/overall": 0.2, "score/vs careful_ai": 0.1, "move loss/training": 2.0},
            {"round": 2, "seconds": 6.0, "score/overall": 0.4, "move loss/training": 1.5},
        ]
        groups = group_metrics(rounds)
        self.assertEqual(list(groups), ["score", "move loss"])
        self.assertEqual(groups["score"]["overall"], [(1, 0.2), (2, 0.4)])
        # A metric missing from a round simply has fewer points.
        self.assertEqual(groups["score"]["vs careful_ai"], [(1, 0.1)])
        self.assertNotIn("seconds", groups)

    def test_a_group_logged_only_sometimes_keeps_its_place(self):
        rounds = [
            {"round": 1, "seconds": 1.0, "entropy/of the policy": 4.0},
            {"round": 2, "seconds": 2.0, "score/overall": 0.3, "entropy/of the policy": 3.9},
            {"round": 3, "seconds": 3.0, "entropy/of the policy": 3.8},
        ]
        self.assertEqual(list(group_metrics(rounds)), ["score", "entropy"])

    def test_share_charts_always_span_zero_to_one(self):
        self.assertEqual(value_axis([0.4, 0.45], "score"), (0.0, 1.0, [0.0, 0.25, 0.5, 0.75, 1.0]))

    def test_other_charts_get_round_numbers_that_cover_the_data(self):
        for values in ([2.21, 2.91], [0.293, 0.55], [1200.0, 5400.0], [3.0, 3.0], [0.0011, 0.0019]):
            low, high, ticks = value_axis(values, "move loss")
            self.assertLessEqual(low, min(values))
            self.assertGreaterEqual(high, max(values))
            self.assertGreaterEqual(len(ticks), 2)
            self.assertLessEqual(len(ticks), 7)
            steps = {round(b - a, 9) for a, b in zip(ticks, ticks[1:])}
            self.assertEqual(len(steps), 1)

    def test_round_ticks_stay_few_and_whole(self):
        self.assertEqual(round_ticks(1, 1), [1])
        self.assertEqual(round_ticks(1, 5), [1, 2, 3, 4, 5])
        for last in (20, 137, 1000, 4321):
            ticks = round_ticks(1, last)
            self.assertLessEqual(len(ticks), 9)
            self.assertTrue(all(1 <= tick <= last for tick in ticks))

    def test_values_are_formatted_for_their_chart(self):
        self.assertEqual(format_value(0.256, "score"), "26%")
        self.assertEqual(format_value(2.2143, "move loss"), "2.21")
        self.assertEqual(format_value(12345.6, "games"), "12,346")


class MonitorWindowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for module in (ml.training.runlog, ml.training.monitor):
            patcher = mock.patch.object(module, "RUNS_DIRECTORY", self.root)
            patcher.start()
            self.addCleanup(patcher.stop)

    def log_rounds(self, count, series=3):
        with mock.patch("builtins.print"):
            log = RunLog("demo", {})
            for number in range(1, count + 1):
                metrics = {f"score/vs player {index}": min(1.0, 0.1 * index + 0.02 * number) for index in range(series)}
                metrics["move loss/training"] = 3.0 / number
                log.log(number, metrics, "")

    def test_draws_before_the_run_exists_and_before_the_first_round(self):
        monitor = Monitor("demo")
        monitor.refresh()
        monitor.draw((0, 0))
        self.assertEqual(monitor.rounds, [])

        self.log_rounds(0)
        monitor.refresh()
        monitor.draw((0, 0))

    def test_picks_up_new_rounds_while_open(self):
        self.log_rounds(1)
        monitor = Monitor("demo")
        monitor.refresh()
        monitor.draw((300, 300))
        self.assertEqual(len(monitor.rounds), 1)

        with mock.patch("builtins.print"), (self.root / "demo" / "metrics.jsonl").open("a") as file:
            file.write('{"round": 2, "seconds": 9.0, "score/vs player 0": 0.5, "move loss/training": 1.0}\n')
        os.utime(self.root / "demo" / "metrics.jsonl", (1, 2_000_000_000))
        monitor.refresh()
        self.assertEqual(len(monitor.rounds), 2)

    def test_charts_tables_and_hover_all_draw(self):
        self.log_rounds(40, series=6)  # more lines than colours: the extra ones are left out, not recoloured
        monitor = Monitor("demo")
        monitor.refresh()
        before = monitor.screen.copy()
        for mouse in ((-50, -50), (300, 300), (900, 300), (1190, 810)):
            monitor.draw(mouse)
        self.assertNotEqual(bytes(before.get_view("1")), bytes(monitor.screen.get_view("1")))
        monitor.show_tables = True
        monitor.draw((300, 300))


if __name__ == "__main__":
    unittest.main()
