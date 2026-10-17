#!/bin/sh
# Step 4, stage 2 (2026-10-12): does a look-ahead help free_gap0 round 50 at all?
#
# Beast is that network on its own at temperature 0.2. Two look-aheads built on the same network
# (its policy proposes the 8 likeliest routes for each side, each at its likeliest amount; `nash`)
# play Beast, 200 games each, on the comparisons' maps 900300-900399, both seats: the same maps
# as Beast's own deciding match. They differ only in who judges the table:
#   1. lookahead_own:    the network's own value output
#   2. lookahead_judge1: judge1 (trained on games of Neural AI's network; reads the old inputs)
# Each shows its score live, game by game; results in ml/runs/logs/beast_match_<label>_maps300.json.
# About 25 minutes each.
#
# Run from the game directory:  sh ml/scripts/judge4_stage2.sh
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
NETWORK=ml/runs/free_gap0/round_0050.pt

$PY ml/scripts/beast_match.py --label lookahead_own --player "lookahead:$NETWORK?rule=nash" || exit 1
$PY ml/scripts/beast_match.py --label lookahead_judge1 --player "lookahead:$NETWORK?rule=nash&judge=ml/runs/judge1/best.pt" || exit 1
