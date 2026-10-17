import base64
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import ml.colab as colab


class ColabTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "ml" / "runs" / "old").mkdir(parents=True)
        (self.root / "ml" / "tests").mkdir()
        (self.root / "ml" / "play" / "__pycache__").mkdir(parents=True)
        (self.root / "ais").mkdir()
        (self.root / "maps").mkdir()
        for name, text in {
            "game.py": "", "ml/__init__.py": "", "ml/play/rollout.py": "", "ml/play/__pycache__/rollout.pyc": "x",
            "ml/tests/test_x.py": "", "ml/runs/old/round_0001.pt": "w", "ml/runs/start.pt": "w",
            "ais/strategic_ai.py": "", "maps/ring.json": "{}",
        }.items():
            (self.root / name).write_text(text)
        for patch in (mock.patch.object(colab, "GAME_ROOT", self.root), mock.patch.object(colab, "FOLDER", self.root / "ml" / "colab")):
            patch.start()
            self.addCleanup(patch.stop)

    def test_the_bundle_takes_the_code_and_what_is_asked_for_and_nothing_else(self):
        names = set(zipfile.ZipFile(io.BytesIO(colab.bundle(["ml/runs/start.pt"]))).namelist())
        self.assertEqual(
            names,
            {"game.py", "ml/__init__.py", "ml/play/rollout.py", "ais/strategic_ai.py", "maps/ring.json", "ml/runs/start.pt"},
        )

    def test_a_run_notebook_gives_back_its_output_and_files(self):
        path = colab.build("demo", ["python -m ml benchmark"], [], ["ml/runs/new"])
        notebook = json.loads(path.read_text())
        self.assertIn("python -m ml benchmark", "".join(notebook["cells"][3]["source"]))
        # As if it had run on Colab: some output, then a run folder sent back.
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("ml/runs/new/metrics.jsonl", '{"round": 1}\n')
        sent = f"{colab.BRING_BACK_START}\n{base64.encodebytes(buffer.getvalue()).decode()}{colab.BRING_BACK_END}\n"
        notebook["cells"][3]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["timing: 12 s\n"]}]
        notebook["cells"][4]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": [sent]}]
        path.write_text(json.dumps(notebook))
        text, unpacked = colab.read("demo")
        self.assertIn("timing: 12 s", text)
        self.assertIn("[1 files sent back]", text)
        self.assertEqual(unpacked, ["ml/runs/new/metrics.jsonl"])
        self.assertEqual((self.root / "ml/runs/new/metrics.jsonl").read_text(), '{"round": 1}\n')
        with self.assertRaises(SystemExit):
            colab.read("demo")  # would overwrite what is there now

    def test_a_long_run_notebook_starts_the_script_detached_watches_it_and_sends_back(self):
        (self.root / "ml" / "scripts").mkdir()
        (self.root / "ml" / "scripts" / "long.sh").write_text("echo all done\n")
        path = colab.build_run("long", "ml/scripts/long.sh", [], ["ml/runs/long"])
        cells = ["".join(cell["source"]) for cell in json.loads(path.read_text())["cells"]]
        packed = cells[1].split('BUNDLE = """\n', 1)[1].split('"""', 1)[0]
        self.assertIn("ml/scripts/long.sh", zipfile.ZipFile(io.BytesIO(base64.b64decode(packed))).namelist())
        self.assertTrue(any("drive.mount" in cell for cell in cells))
        start = next(cell for cell in cells if "start_new_session=True" in cell)
        self.assertIn('"pgrep", "-f", "long.sh"', start)
        self.assertTrue(any("time.sleep(600)" in cell for cell in cells))
        self.assertIn(colab.BRING_BACK_START, cells[-1])


if __name__ == "__main__":
    unittest.main()
