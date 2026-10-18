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
    "train": ("ml.training.train", "Train a network: imitation, selfplay or kickstart"),
    "monitor": ("ml.training.monitor", "Live charts for a training run"),
    "tensorboard": ("ml.training.board", "Training curves of one or several runs in TensorBoard, in the browser"),
    "evaluate": ("ml.evaluation.evaluate", "Play a player against opponents on fresh maps"),
    "compare": ("ml.evaluation.compare", "Rate the saved rounds of runs against each other"),
    "sweep": ("ml.evaluation.sweep", "Train several settings and compare the results"),
    "judge-games": ("ml.judge.games", "Judge loop, step 2: play games and record labelled positions"),
    "judge-train": ("ml.judge.train", "Judge loop, step 3: train the judge on recorded positions"),
    "judge-check": ("ml.judge.check", "Judge loop, step 4: test the look-ahead with the new judge"),
    "mistakes": ("ml.evaluation.mistakes", "Where a network fails to copy strategic_ai, move by move"),
    "policy-check": ("ml.judge.policy_check", "A network with the look-ahead against Beast and Atlas, and on instinct against another"),
    "beast": ("ml.play.beast", "Show or change who Beast AI (our strongest player) is"),
    "rematch": ("ml.evaluation.rematch", "Play a saved game's start again many times and count"),
    "watch": ("ml.evaluation.watch", "Watch a saved game in the game window"),
    "summary": ("ml.evaluation.summary", "Put the final numbers of several runs side by side"),
    "lineage": ("ml.evaluation.lineage", "One set of training curves for a network trained over several runs"),
    "behaviour": ("ml.evaluation.behaviour", "How networks play: where sends go, how decided the amounts are"),
    "bot-ratings": ("ml.evaluation.bot_ratings", "Rate the AIs in the game's menu for the home screen (ais/ratings.json)"),
    "benchmark": ("ml.evaluation.benchmark", "How fast this machine trains: playing and learning, timed"),
    "colab": ("ml.colab", "Run ml commands on Colab through a notebook that carries the code (build, read)"),
    "rollout": ("ml.play.rollout", "Play many games at once and report speed"),
    "checkpoint": ("ml.models.checkpoint", "Create or inspect a checkpoint"),
    "maps": ("ml.map_generator.viewer", "Look through generated maps"),
    "show-state": ("ml.encoding.show_state", "Print what the network sees for one position"),
    "show-actions": ("ml.encoding.show_actions", "Print the moves the network can choose from"),
    "show-model": ("ml.models.show_model", "Print what a model makes of one position"),
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
