"""The learned agent from ml/, playable like any other AI.

It plays the checkpoint named by the NEURAL_AI_CHECKPOINT environment
variable, or ml/runs/latest.pt if that is not set, or, in a checkout without
training runs, the network committed in checkpoints/neural_ai.pt. With no
checkpoint at all it plays an untrained network, which moves legally but at random.

    NEURAL_AI_CHECKPOINT=ml/runs/demo/random.pt .venv/bin/python main.py --red-ai neural_ai

The file is checked again at the start of every match, so while a training
run keeps saving to it, each new match shows the newest version. Without
PyTorch installed, it plays the same network through numpy (ml/play/numpy_agent.py).

A network from a run trained under a send limit (``--min-send-gap``) plays
under that limit here too, as it does in every measurement:

    NEURAL_AI_CHECKPOINT=ml/runs/send_gap10/round_0150.pt .venv/bin/python main.py

Its moves are drawn by chance at TEMPERATURE: below 1, its likelier moves come up more
often than in training, so it plays its favourite more often, but not always.
"""

import os
import sys
from pathlib import Path


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "ml" / "runs" / "latest.pt"
COMMITTED_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "neural_ai.pt"
# 0.2 (the user's choice, 2026-10-10): it plays its favourite route about 72% of the time,
# against 28% drawing by the network's own odds (1.0), which shuffled units back and forth far
# more and lost 33.5% to 66.5% to always playing the favourite. 0.8 changed too little (33%).
# See ml/history.md, "Playing the favourite instead of drawing by chance".
TEMPERATURE = 0.2

_agent = None
_loaded_version = None


def choose_action(state):
    global _agent, _loaded_version
    if _agent is None or state.tick == 0:
        default = DEFAULT_CHECKPOINT if DEFAULT_CHECKPOINT.exists() else COMMITTED_CHECKPOINT
        path = Path(os.environ.get("NEURAL_AI_CHECKPOINT", default))
        version = (path, path.stat().st_mtime) if path.exists() else (path, None)
        if _agent is None or version != _loaded_version:
            _agent = _load_agent(path)
            _loaded_version = version
    return _agent.choose_action(state)


def _load_agent(path):
    # Imported here so that choosing another AI never loads PyTorch. Without PyTorch (as on the
    # Arena website, which plays in the browser) the same network plays through numpy.
    try:
        from ml.play.agent import NeuralAgent
        from ml.training.runlog import min_send_gap
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        from ml.play.numpy_agent import NumpyAgent as NeuralAgent, min_send_gap

    if path.exists():
        gap = min_send_gap(path)
        if gap > 1:
            print(f"neural_ai: {path} trained with at most one send every {gap} ticks, and plays so", file=sys.stderr)
        return NeuralAgent.from_init(path, min_send_gap=gap, temperature=TEMPERATURE)
    print(f"neural_ai: no checkpoint at {path}, playing an untrained network", file=sys.stderr)
    return NeuralAgent.from_init("random", seed=0)
