#!/bin/sh
# Atlas Clean's strength: against the scripted AIs, overnight round 500, the original Atlas, and
# itself planning all at once (to see what spreading the planning over the ticks costs).
#
# 16 fresh maps from both seats, so 32 games per pairing. About 45 minutes: games involving the
# original Atlas are slow, because against an opponent it does not recognise it simulates with a
# network (pinned here to round 500, so the run cannot change what it uses).
#
# Run:  ml/scripts/done/atlas_clean_check.sh
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
NET=ml/runs/overnight_scratch/round_0500.pt
ATLAS_OPPONENT_CHECKPOINT=$NET $PY -u -m ml evaluate --candidate atlas_clean_ai \
    --opponents aggressive_ai careful_ai strategic_ai "$NET" atlas_ai _atlas_clean_blocking \
    --generated-maps 16 --no-bundled-maps --skip-generated 30 || exit 1
echo "=== atlas clean check finished"
