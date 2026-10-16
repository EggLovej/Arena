# Atlas opponent-model provenance

This inference-only package is adapted from the existing game ML implementation.
Imports are relocated from ml to atlas_model; the original ml directory and
its live training files are untouched. The current graph-transformer route
padding implementation was copied on 2026-10-01. The optional `route_layers`
setting (default 1, extra hidden layers in the route head) was copied the same
day, so checkpoints saved by ml/ after that change still load here.

The bundled opponent.pt is the existing league checkpoint: 120,120 weights,
600 strategic_ai imitation games, then 30 league rounds / 2,880 games, saved
2026-09-30 17:59:32 local time. Its SHA-256 is
1dc32ba89e8dfaed04be0d76e8e475b2fcc97d9762032d545476cf9e91ba42d6.
Those weights were produced by the separate ML work, not trained by Atlas.

Atlas uses the model to forecast opponent action probabilities. It prefers
ATLAS_OPPONENT_CHECKPOINT, then NEURAL_AI_CHECKPOINT, then ml/runs/latest.pt,
falling back to this bundled file. It pins independent bytes for the match.
No running opponent object or random generator is accessed. Loading preserves
the process's Torch RNG. NumPy and PyTorch are required for learned planning.
