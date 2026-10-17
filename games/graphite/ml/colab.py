"""Run ml commands on a Colab runtime, through a notebook that carries everything it needs.

Colab only runs notebooks, and its runtime starts empty. So ``build`` packs
the game, ``ml/`` and any checkpoints into the notebook's first cell; the
notebook unpacks them on the runtime and runs the commands, one cell each.
Results come back the same way, inside the notebook: the commands' output,
and, if asked, files (e.g. a run folder) packed into the last cell's output.
Once the notebook has run and is saved, ``read`` prints the output and
unpacks the files into this folder.

    .venv/bin/python -m ml colab build --name benchmark --command "python -u -m ml benchmark" \\
        --include ml/runs/shaping_entropy/round_0150.pt
    (open ml/colab/benchmark.ipynb in VS Code, pick the Colab runtime, Run All, save)
    .venv/bin/python -m ml colab read --name benchmark

For long runs, ``run`` writes a notebook that starts a script detached on the
runtime and keeps the run on Google Drive (see ``build_run``):

    .venv/bin/python -m ml colab run --name scratch_256 --script ml/scripts/colab_scratch_256.sh \\
        --bring-back ml/runs/compare_scratch_256 ml/runs/scratch_256/metrics.jsonl

Notebooks are written to ``ml/colab/``.
"""

import argparse
import base64
import io
import json
import re
import zipfile
from pathlib import Path

FOLDER = Path("ml/colab")
GAME_ROOT = Path(".")
# What the runtime needs: the game's own files, its AIs and maps, and ml/ without runs and tests.
SKIPPED_PARTS = {"__pycache__", "runs", "tests", "colab"}
BRING_BACK_START, BRING_BACK_END = "=== FILES BACK START ===", "=== FILES BACK END ==="


def bundle(include: list[str], with_atlas: bool = False) -> bytes:
    """The code and the named extra files, as one zip."""

    paths = [path for path in GAME_ROOT.glob("*.py")]
    folders = ["ais", "maps", "ml"] + (["atlas_model"] if with_atlas else [])
    for folder in folders:
        for path in (GAME_ROOT / folder).rglob("*"):
            folders_on_the_way = set(path.relative_to(GAME_ROOT).parts[:-1])
            if path.is_file() and path.suffix in (".py", ".json", ".sh") and not SKIPPED_PARTS & folders_on_the_way:
                paths.append(path)
    for item in include:
        path = GAME_ROOT / item
        if not path.exists():
            raise SystemExit(f"{item} does not exist")
        paths += [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            archive.write(path, path.relative_to(GAME_ROOT).as_posix())
    return buffer.getvalue()


def _code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source.splitlines(keepends=True)}


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def build(name: str, commands: list[str], include: list[str], bring_back: list[str], with_atlas: bool = False) -> Path:
    packed = base64.encodebytes(bundle(include, with_atlas)).decode()
    unpack = (
        '"""Unpack the game and ml/ onto this runtime (packed by ml colab build)."""\n'
        "import base64, io, os, zipfile\n"
        f'BUNDLE = """\n{packed}"""\n'
        'ROOT = "/content/agent_game" if os.path.isdir("/content") else os.path.expanduser("~/agent_game")\n'
        "archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE)))\n"
        "archive.extractall(ROOT)\n"
        "os.chdir(ROOT)\n"
        'print(f"unpacked {len(archive.namelist())} files into {ROOT}")\n'
    )
    check = (
        '"""What this runtime has."""\n'
        "import os, platform, torch\n"
        'print("python", platform.python_version(), "| torch", torch.__version__, "| cpu cores", os.cpu_count())\n'
        'print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")\n'
    )
    cells = [
        _markdown(
            f"# {name}\n\n"
            "Built by `ml colab build`. Pick a Colab runtime as the kernel, **Run All**, and when it has "
            "finished **save the notebook**: the output (and any files it sends back) is stored in this "
            f"file, and `ml colab read --name {name}` takes it from there.\n\n"
            "Commands:\n\n" + "\n".join(f"- `{command}`" for command in commands)
        ),
        _code(unpack),
        _code(check),
    ]
    for command in commands:
        cells.append(_code(f"!cd /content/agent_game 2>/dev/null || cd ~/agent_game; {command}\n"))
    if bring_back:
        cells.append(
            _code(
                '"""Send files back: packed into this cell\'s output, which is saved with the notebook."""\n'
                "import base64, io, os, zipfile\n"
                f"PATHS = {bring_back!r}\n"
                "files = []\n"
                "for item in PATHS:\n"
                "    if os.path.isdir(item):\n"
                "        files += [os.path.join(folder, name) for folder, _, names in os.walk(item) for name in names]\n"
                "    elif os.path.exists(item):\n"
                "        files.append(item)\n"
                "buffer = io.BytesIO()\n"
                'with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:\n'
                "    for file in files:\n"
                "        archive.write(file)\n"
                f'print("{BRING_BACK_START}")\n'
                "print(base64.encodebytes(buffer.getvalue()).decode(), end=\"\")\n"
                f'print("{BRING_BACK_END}")\n'
            )
        )
    cells.append(
        _markdown(
            "Optional, for long runs later: can this runtime reach Google Drive (a place for run folders "
            "that survives the runtime being reset)? Run the next cell only if you want to find out."
        )
    )
    cells.append(_code('import os\nfrom google.colab import drive\ndrive.mount("/content/drive")\nprint(os.listdir("/content/drive"))\n'))
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    FOLDER.mkdir(parents=True, exist_ok=True)
    path = FOLDER / f"{name}.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    return path


DRIVE = "/content/drive/MyDrive/agent_game"  # where long runs keep their files on the user's Google Drive


def build_run(name: str, script: str, include: list[str], bring_back: list[str], with_atlas: bool = False) -> Path:
    """A notebook for a long run: it starts ``script`` detached on the runtime, keeps the run on
    Google Drive (the script copies its files there), shows progress on demand, and sends the
    results back at the end. Running it again on a new runtime carries on where it stopped."""

    if not (GAME_ROOT / script).exists():
        raise SystemExit(f"{script} does not exist")
    packed = base64.encodebytes(bundle(include + [script], with_atlas)).decode()
    log = f"{DRIVE}/{name}_log.txt"
    cells = [
        _markdown(
            f"# {name}: a long run on Colab\n\n"
            f"Runs `{script}` (built by `ml colab run`). **Run All** once: it unpacks the code, connects "
            "Google Drive (sign in when asked), and starts the run in the background on the runtime; "
            f"the run keeps its files on Drive in `MyDrive/agent_game/`.\n\n"
            "- **While it runs:** the *watch* cell prints the latest line every 10 minutes and ends when "
            "the run is done; then the last cell packs the results. Keep VS Code connected and the Mac "
            "awake meanwhile. The *progress* cell shows more lines whenever you run it.\n"
            "- **If the runtime was reset or disconnected:** Run All again; the run carries on from its "
            "last 50-round checkpoint on Drive.\n"
            "- **When it is done:** **save the notebook**, and "
            f"`ml colab read --name {name}` unpacks the results on the Mac."
        ),
        _code(
            '"""Unpack the game and ml/ onto this runtime (packed by ml colab run)."""\n'
            "import base64, io, os, zipfile\n"
            f'BUNDLE = """\n{packed}"""\n'
            'ROOT = "/content/agent_game"\n'
            "archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE)))\n"
            "archive.extractall(ROOT)\n"
            "os.chdir(ROOT)\n"
            'print(f"unpacked {len(archive.namelist())} files into {ROOT}")\n'
        ),
        _code(
            '"""What this runtime has."""\n'
            "import os, platform, torch\n"
            'print("python", platform.python_version(), "| torch", torch.__version__, "| cpu cores", os.cpu_count())\n'
            'print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")\n'
        ),
        _code(
            '"""Google Drive: where the run keeps its files, so it survives the runtime being reset."""\n'
            "import os\n"
            "from google.colab import drive\n"
            'drive.mount("/content/drive")\n'
            f'os.makedirs("{DRIVE}", exist_ok=True)\n'
            f'print("run files go to {DRIVE}")\n'
        ),
        _code(
            '"""Start the run in the background (or report that it is already running)."""\n'
            "import os, subprocess\n"
            f'running = subprocess.run(["pgrep", "-f", "{Path(script).name}"], capture_output=True).returncode == 0\n'
            "if running:\n"
            '    print("already running; see the progress cell")\n'
            "else:\n"
            f'    log = open("{log}", "a")\n'
            f'    subprocess.Popen(["sh", "{script}"], cwd="/content/agent_game", stdout=log, stderr=subprocess.STDOUT,\n'
            f'                     env={{**os.environ, "DRIVE": "{DRIVE}"}}, start_new_session=True)\n'
            f'    print("started; log: {log}")\n'
        ),
        _code(
            '"""Progress: run this cell again whenever you want to look."""\n'
            "import json, os\n"
            f'log = "{log}"\n'
            'lines = open(log).read().splitlines() if os.path.exists(log) else []\n'
            'print("\\n".join(line[:220] for line in lines[-12:]) or "no log yet")\n'
        ),
        _code(
            '"""Watch: the latest line every 10 minutes, until the run is done. A running cell keeps\n'
            "Colab from treating the runtime as idle while the run works in the background. Stopping\n"
            'this cell does not stop the run."""\n'
            "import os, time\n"
            f'log = "{log}"\n'
            "while True:\n"
            '    text = open(log).read() if os.path.exists(log) else ""\n'
            '    last = text.strip().splitlines()[-1][:200] if text.strip() else "no log yet"\n'
            '    print(time.strftime("%H:%M"), last, flush=True)\n'
            '    if "all done" in text or "Traceback" in text[-5000:]:\n'
            "        break\n"
            "    time.sleep(600)\n"
        ),
    ]
    if bring_back:
        cells.append(
            _code(
                '"""Send the results back: packed into this cell\'s output, saved with the notebook."""\n'
                "import base64, io, os, zipfile\n"
                f"PATHS = {bring_back!r}\n"
                f'if "all done" not in open("{log}").read():\n'
                '    print("not finished yet: run this cell again once the progress cell says all done")\n'
                "else:\n"
                "    files = []\n"
                "    for item in PATHS:\n"
                "        # The runtime's own copy if it has one, else the one the run kept on Drive.\n"
                f'        drive_copy = os.path.join("{DRIVE}/runs", os.path.relpath(item, "ml/runs"))\n'
                '        source = os.path.join("/content/agent_game", item)\n'
                "        source = source if os.path.exists(source) else drive_copy\n"
                "        if os.path.isdir(source):\n"
                "            files += [(os.path.join(f, n), os.path.join(item, os.path.relpath(os.path.join(f, n), source)))\n"
                "                      for f, _, names in os.walk(source) for n in names]\n"
                "        elif os.path.exists(source):\n"
                "            files.append((source, item))\n"
                "    buffer = io.BytesIO()\n"
                '    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:\n'
                "        for source, name in files:\n"
                "            archive.write(source, name)\n"
                f'    print("{BRING_BACK_START}")\n'
                '    print(base64.encodebytes(buffer.getvalue()).decode(), end="")\n'
                f'    print("{BRING_BACK_END}")\n'
            )
        )
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    FOLDER.mkdir(parents=True, exist_ok=True)
    path = FOLDER / f"{name}.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    return path


def read(name: str, overwrite: bool = False) -> tuple[str, list[str]]:
    """The commands' output from a run notebook, and the files it sent back (unpacked here)."""

    notebook = json.loads((FOLDER / f"{name}.ipynb").read_text())
    texts = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                texts.append("".join(output.get("text", [])))
            elif "data" in output and "text/plain" in output["data"]:
                texts.append("".join(output["data"]["text/plain"]))
    text = "".join(texts)
    unpacked = []
    match = re.search(re.escape(BRING_BACK_START) + r"\n(.*?)" + re.escape(BRING_BACK_END), text, re.S)
    if match:
        archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(match.group(1))))
        for member in archive.namelist():
            target = GAME_ROOT / member
            if target.exists() and not overwrite:
                raise SystemExit(f"{target} exists already; read again with --overwrite to replace it")
        archive.extractall(GAME_ROOT)
        unpacked = archive.namelist()
        text = text[: match.start()] + f"[{len(unpacked)} files sent back]\n" + text[match.end() :]
    return text, unpacked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    actions = parser.add_subparsers(dest="action", required=True)
    make = actions.add_parser("build", help="Write ml/colab/NAME.ipynb")
    make.add_argument("--name", required=True)
    make.add_argument("--command", action="append", required=True, help="A shell command to run there, in the game folder; repeat for more")
    make.add_argument("--include", nargs="*", default=[], help="Files or folders to take along, e.g. a starting checkpoint")
    make.add_argument("--bring-back", nargs="*", default=[], help="Files or folders to send back in the notebook, e.g. ml/runs/NAME")
    make.add_argument("--with-atlas", action="store_true", help="Also take Atlas's model code (only for games against Atlas)")
    long = actions.add_parser("run", help="Write ml/colab/NAME.ipynb for a long run kept on Google Drive")
    long.add_argument("--name", required=True)
    long.add_argument("--script", required=True, help="The run's shell script, e.g. ml/scripts/colab_scratch_256.sh")
    long.add_argument("--include", nargs="*", default=[])
    long.add_argument("--bring-back", nargs="*", default=[])
    long.add_argument("--with-atlas", action="store_true")
    take = actions.add_parser("read", help="Print a run notebook's output and unpack the files it sent back")
    take.add_argument("--name", required=True)
    take.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.action in ("build", "run"):
        if args.action == "build":
            path = build(args.name, args.command, args.include, args.bring_back, args.with_atlas)
        else:
            path = build_run(args.name, args.script, args.include, args.bring_back, args.with_atlas)
        print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB). Open it in VS Code, pick the Colab runtime, Run All, then save.")
    else:
        text, unpacked = read(args.name, args.overwrite)
        print(text)
        if unpacked:
            print(f"Unpacked {len(unpacked)} files, e.g. {unpacked[0]}")


if __name__ == "__main__":
    main()
