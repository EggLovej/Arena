"""One entry point for every ml command.

Run from the game directory:

    .venv/bin/python -m ml                      list the commands
    .venv/bin/python -m ml train selfplay --run demo
    .venv/bin/python -m ml evaluate --help

Everything after the command name goes to that command unchanged.
"""

import importlib
import sys


# command name -> (module, one line for the list)
COMMANDS = {
    "train": ("ml.training.train", "Train a network: imitation or selfplay"),
    "tensorboard": ("ml.training.board", "Training curves of one or several runs in TensorBoard, in the browser"),
    "evaluate": ("ml.evaluation.evaluate", "Play a player against opponents on fresh maps"),
    "compare": ("ml.evaluation.compare", "Rate the saved rounds of runs against each other"),
    "beast": ("ml.play.beast", "Show or change who Beast AI (our strongest player) is"),
    "rollout": ("ml.play.rollout", "Play many games at once and report speed"),
    "checkpoint": ("ml.models.checkpoint", "Create or inspect a checkpoint"),
    "maps": ("ml.map_generator.viewer", "Look through generated maps"),
}


def print_commands() -> None:
    print(__doc__.splitlines()[0] + "\n")
    print("usage: python -m ml <command> [options]\n")
    width = max(len(name) for name in COMMANDS)
    for name, (_, summary) in COMMANDS.items():
        print(f"  {name:{width}}  {summary}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print_commands()
        return
    name = sys.argv[1]
    if name not in COMMANDS:
        print(f"unknown command {name!r}\n", file=sys.stderr)
        print_commands()
        sys.exit(2)
    module = importlib.import_module(COMMANDS[name][0])
    # The command parses sys.argv itself; this makes its usage line read "ml <command>".
    sys.argv = [f"ml {name}", *sys.argv[2:]]
    module.main()


# Worker processes import this file again under another name; only the
# process that was started from the command line may run a command.
if __name__ == "__main__":
    main()
