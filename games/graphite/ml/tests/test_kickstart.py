import io
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import ml.training.runlog
from ml.encoding.action_encoder import AMOUNT_OPTIONS
from ml.models import build_model, make_input
from ml.models.amounts import on_grid
from ml.models.checkpoint import load_checkpoint, save_checkpoint
from ml.tests.test_models import positions
from ml.training.kickstart import KickstartSettings, free_amount_divergence, target_amounts, train_kickstart
from ml.training.runlog import read_metrics


class TargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        states, legal = positions(seeds=range(4), every=20, ticks=400)
        cls.inputs = make_input(states, legal)
        # A teacher that knows only the first options, as if trained before the others existed.
        cls.known = len(AMOUNT_OPTIONS) - 2
        with torch.no_grad():
            cls.taught = build_model(amount_options=cls.known).eval()(cls.inputs)

    def test_new_options_on_offer_share_a_little_and_the_rest_keeps_its_proportions(self):
        target = target_amounts(self.taught, self.inputs.allowed, self.known, share=0.1)
        teacher = torch.softmax(self.taught.amount_logits, dim=-1)
        usable = self.inputs.allowed.any(dim=-1)
        offered = self.inputs.allowed[..., self.known :].any(dim=-1) & usable
        self.assertTrue(offered.any(), "the positions should offer some new option somewhere")
        self.assertTrue(torch.allclose(target[usable].sum(dim=-1), torch.ones(int(usable.sum())), atol=1e-5))
        self.assertTrue(torch.allclose(target[offered][:, self.known :].sum(dim=-1), torch.full((int(offered.sum()),), 0.1), atol=1e-5))
        self.assertTrue(torch.allclose(target[offered][:, : self.known], 0.9 * teacher[offered][:, : self.known], atol=1e-6))
        # Where no new option is on offer, the teacher's probabilities stay as they are.
        plain = usable & ~offered
        self.assertTrue(torch.allclose(target[plain], teacher[plain]))
        # And a new option that is not on offer gets nothing.
        self.assertEqual(target[usable][~self.inputs.allowed[usable]].max().item(), 0.0)


class FreeStudentTests(unittest.TestCase):
    def test_the_divergence_counts_options_that_send_the_same_units_as_one_amount(self):
        torch.manual_seed(0)
        states, legal = positions(seeds=range(3), every=40, ticks=300)
        inputs = make_input(states, legal)
        bases = inputs.base_mask.shape[1]
        units = torch.zeros(*inputs.allowed.shape, dtype=torch.long)
        for row, entry in enumerate(legal):
            count = len(entry.base_ids)
            units[row, :count, :count] = torch.from_numpy(entry.units)
        with torch.no_grad():
            taught = build_model().eval()(inputs)
            # Curves of every shape, not only the ones a fresh network starts with.
            learned = build_model(amounts="free").eval()(inputs)
            learned = replace(learned, amount_curves=torch.randn(learned.amount_curves.shape, generator=torch.Generator().manual_seed(1)) * 2)
            divergence = free_amount_divergence(taught, learned, units)

        checked = merged = 0
        for row, source, target in zip(*np.nonzero(inputs.most.numpy())):
            most = int(inputs.most[row, source, target])
            teacher = torch.softmax(taught.amount_logits[row, source, target], dim=0)
            by_units = {}
            for option, sent in enumerate(units[row, source, target].tolist()):
                if sent > 0:
                    by_units[sent] = by_units.get(sent, 0.0) + float(teacher[option])
            merged += len(by_units) < int((units[row, source, target] > 0).sum())
            # In double precision, so that even very unlikely amounts keep a probability above 0.
            student = on_grid(
                learned.amount_curves[row, source, target][None].double(), inputs.most[row, source, target][None].double(),
                inputs.needed[row, source, target][None].double(), most,
            )[0]
            expected = sum(p * (np.log(p) - np.log(float(student[sent - 1]))) for sent, p in by_units.items() if p > 0)
            self.assertLess(abs(float(divergence[row, source, target]) - expected), 1e-3 * max(1.0, expected))
            checked += 1
        self.assertGreater(checked, 50)
        self.assertGreater(merged, 5, "some routes should have options that send the same units")
        # Nothing on routes that cannot be used.
        self.assertEqual(float(divergence[inputs.most == 0].abs().max()), 0.0)
        self.assertEqual(divergence.shape, (len(states), bases, bases))


class KickstartRunTests(unittest.TestCase):
    def test_a_small_run_learns_the_teacher_and_leaves_a_run_folder(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(ml.training.runlog, "RUNS_DIRECTORY", Path(directory)):
            teacher = save_checkpoint(Path(directory) / "teacher.pt", build_model(layers=2, amount_options=len(AMOUNT_OPTIONS) - 2))
            student = save_checkpoint(Path(directory) / "student.pt", build_model(width=32, layers=2, capture_head=True))
            settings = KickstartSettings(
                teacher=str(teacher), games=12, decisions_per_game=60, held_out_share=0.25, epochs=3,
                patience=0, batch_size=64, workers=2,
            )
            with redirect_stdout(io.StringIO()) as output:
                best = train_kickstart("small", settings, init=str(student), device="cpu")
            metrics = read_metrics("small")
            learned = load_checkpoint(best)
            self.assertTrue(best.exists())
        self.assertEqual([entry["round"] for entry in metrics], [0, 1, 2, 3])
        self.assertLess(metrics[-1]["divergence/held out"], metrics[0]["divergence/held out"])
        self.assertIn("capture error/held out", metrics[-1])
        self.assertIn("same favourite", output.getvalue())
        self.assertTrue(learned.model.config.capture_head)
        self.assertEqual(learned.history[-1]["step"], "kickstarted")

    def test_a_student_with_free_amounts_learns_from_a_teacher_with_options(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(ml.training.runlog, "RUNS_DIRECTORY", Path(directory)):
            teacher = save_checkpoint(Path(directory) / "teacher.pt", build_model(layers=2))
            student = save_checkpoint(Path(directory) / "student.pt", build_model(width=32, layers=2, amounts="free", feature_set="v2"))
            settings = KickstartSettings(
                teacher=str(teacher), games=12, decisions_per_game=60, held_out_share=0.25, epochs=3,
                patience=0, batch_size=64, workers=2,
            )
            with redirect_stdout(io.StringIO()):
                best = train_kickstart("free", settings, init=str(student), device="cpu")
            metrics = read_metrics("free")
            learned = load_checkpoint(best)
        self.assertLess(metrics[-1]["divergence/held out"], metrics[0]["divergence/held out"])
        self.assertTrue(learned.model.free_amounts)
        self.assertEqual(learned.model.config.feature_set, "v2")


if __name__ == "__main__":
    unittest.main()
