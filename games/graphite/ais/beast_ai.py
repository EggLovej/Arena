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
    """The game window's player: one Beast for every seat it plays."""

    global _player, _loaded
    if _player is None or state.tick == 0:
        # Imported here so that choosing another AI never loads PyTorch.
        from ml.play.beast import current

        entry = current()
        if _player is None or entry != _loaded:
            _player = _build(entry)
            _loaded = entry
    return _player(state)


def create_bot():
    """The Arena's player: a Beast of its own that draws its moves from the match's seed,
    so a seeded match plays the same way again."""

    return _Bot()


class _Bot:
    def reset(self, context):
        from ml.play.beast import current

        self.player = _build(current(), context.seed)

    def act(self, state):
        return self.player(state)


def _build(entry, seed=0):
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        # Without PyTorch (as on the Arena website, which plays in the browser) a network:
        # Beast plays through numpy; a look-ahead Beast cannot.
        from ml.play.numpy_agent import network_player

        if entry is None:
            raise ValueError("beast_ai: without PyTorch, Beast needs its file (checkpoints/beast.json)")
        return network_player(entry["player"], seed=seed)
    from ml.evaluation.evaluate import load_player

    if entry is None:
        print("beast_ai: no ml/runs/beast.json, playing ml/runs/latest.pt like Neural AI", file=sys.stderr)
        from ais import neural_ai

        return neural_ai.choose_action
    return load_player(entry["player"], seed=seed)
