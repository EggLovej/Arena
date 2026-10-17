"""Beast: our strongest player at any time (see ml/play/beast.py).

Who that is lives in ml/runs/beast.json (or, in a checkout without training
runs, checkpoints/beast.json) and is moved whenever an evaluation shows a
stronger player; the file is checked again at the start of every
match. Today it thinks before every move, so each decision takes a few
dozen milliseconds. Without the file it plays like Neural AI.
"""

import sys

_player = None
_loaded = None  # the entry the current player was built from


def choose_action(state):
    global _player, _loaded
    if _player is None or state.tick == 0:
        # Imported here so that choosing another AI never loads PyTorch.
        from ml.play.beast import current

        entry = current()
        if _player is None or entry != _loaded:
            _player = _build(entry)
            _loaded = entry
    return _player(state)


def _build(entry):
    from ml.evaluation.evaluate import load_player

    if entry is None:
        print("beast_ai: no ml/runs/beast.json, playing ml/runs/latest.pt like Neural AI", file=sys.stderr)
        from ais import neural_ai

        return neural_ai.choose_action
    return load_player(entry["player"], seed=0)
