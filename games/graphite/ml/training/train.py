"""Start a training run.

Run from the game directory:

    .venv/bin/python -m ml train imitation --run copy_the_teacher
    .venv/bin/python -m ml train selfplay --run from_scratch
    .venv/bin/python -m ml train selfplay --run from_teacher --init ml/runs/copy_the_teacher/best.pt
    .venv/bin/python -m ml train league --run my_league --init ml/runs/copy_the_teacher/best.pt
    .venv/bin/python -m ml train kickstart --run bigger --init ml/runs/bigger_start.pt --teacher ml/runs/x/best.pt
    .venv/bin/python -m ml train league --help

Every setting of a trainer is available as an option; ``--help`` lists them
with their defaults. ``--init`` takes "random" or the path of a checkpoint to
continue from. ``--monitor`` opens the live charts next to the run.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import fields

from ml.models import DEFAULT_MODEL, MODELS
from ml.training.imitation import ImitationSettings, train_imitation
from ml.training.kickstart import KickstartSettings, train_kickstart
from ml.training.league import LeagueSettings, train_league
from ml.training.runlog import RUNS_DIRECTORY
from ml.training.selfplay import SelfPlaySettings, train_selfplay


# name -> (settings class, function that runs it)
TRAINERS = {
    "imitation": (ImitationSettings, train_imitation),
    "selfplay": (SelfPlaySettings, train_selfplay),
    "league": (LeagueSettings, train_league),
    "kickstart": (KickstartSettings, train_kickstart),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="trainer", required=True)
    for name, (settings_class, _) in TRAINERS.items():
        command = commands.add_parser(name)
        command.add_argument("--run", required=True, help="Name of the folder in ml/runs/ to write to")
        command.add_argument("--init", default="random", help='"random" (the default) or the path of a checkpoint')
        command.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS), help="Used with --init random")
        command.add_argument("--overwrite", action="store_true", help="Replace an existing run of the same name")
        command.add_argument("--monitor", action="store_true", help="Open the live charts for this run")
        if name not in ("imitation", "kickstart"):
            command.add_argument(
                "--resume",
                action="store_true",
                help="Carry on an existing run from its latest.pt, with the settings it was started with",
            )
            command.add_argument(
                "--stop-after",
                type=int,
                metavar="N",
                help="End this process after N rounds (then continue with --resume in a fresh one)",
            )

        settings = command.add_argument_group("settings")
        for setting in fields(settings_class):
            option = "--" + setting.name.replace("_", "-")
            if isinstance(setting.default, tuple):
                shown = "default: " + " ".join(setting.default)
                settings.add_argument(option, nargs="+", default=list(setting.default), metavar="NAME", help=shown)
            elif isinstance(setting.default, bool):
                shown = f"yes or no; default: {'yes' if setting.default else 'no'}"
                settings.add_argument(option, type=yes_or_no, default=setting.default, metavar="YES/NO", help=shown)
            else:
                shown = f"default: {setting.default}"
                settings.add_argument(option, type=type(setting.default), default=setting.default, metavar="N", help=shown)
    return parser.parse_args()


def yes_or_no(text: str) -> bool:
    if text.lower() in ("yes", "true", "1", "on"):
        return True
    if text.lower() in ("no", "false", "0", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected yes or no, got '{text}'")


def main() -> None:
    args = parse_args()
    settings_class, run_trainer = TRAINERS[args.trainer]
    extra = {}
    if getattr(args, "resume", False):
        # A resumed run keeps the settings it was started with; options on the command line are ignored.
        # A setting added after the run started takes its default, which is how the run behaved.
        stored = json.loads((RUNS_DIRECTORY / args.run / "settings.json").read_text())
        settings = settings_class(
            **{
                setting.name: tuple(stored[setting.name]) if isinstance(setting.default, tuple) else stored[setting.name]
                for setting in fields(settings_class)
                if setting.name in stored
            }
        )
        extra["resume"] = True
    else:
        settings = settings_class(
            **{
                setting.name: tuple(value) if isinstance(value := getattr(args, setting.name), list) else value
                for setting in fields(settings_class)
            }
        )
    if getattr(args, "stop_after", None):
        extra["stop_after"] = args.stop_after

    monitor = None
    if args.monitor:
        monitor = subprocess.Popen([sys.executable, "-m", "ml", "monitor", "--run", args.run, "--wait"])
    try:
        run_trainer(args.run, settings, init=args.init, model_name=args.model, overwrite=args.overwrite, **extra)
    finally:
        if monitor is not None and monitor.poll() is None:
            print("Training has finished. Close the chart window when you are done with it.")


if __name__ == "__main__":
    main()
