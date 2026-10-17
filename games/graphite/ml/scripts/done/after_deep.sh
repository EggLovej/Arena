#!/bin/sh
# After the deep overnight run: Stage 10's comparison, and the final check of our strongest player.
#
# 1. Random start against imitation start, both the 64x6 network with 500 rounds of plain self-play:
#    rounds 0-500 of both on one scale (strategic_ai and the first overnight run's round 500 as the
#    panel, 100 games each), the two final rounds head to head. The imitation run's own 11 minutes
#    count towards its time.
# 2. Move scoring on the deep model (value:deep_scratch round 500, 8 candidates) against the scripted
#    AIs, the plain deep model, the first overnight model, Atlas Clean, and the original Atlas (its
#    opponent network pinned to the deep model, the hardest test). 20 fresh maps, both seats.
#
# Waits for ml/scripts/done/deep_overnight.sh to finish first. About 45 minutes after that.
# Run:  ml/scripts/done/after_deep.sh
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
DEEP=ml/runs/deep_scratch/round_0500.pt

while pgrep -f "[d]eep_overnight[.]sh" > /dev/null; do sleep 60; done
grep -q '"round": 500,' ml/runs/deep_from_imitation/metrics.jsonl || { echo "deep_from_imitation is not finished"; exit 1; }

echo "=== Stage 10: random start against imitation start"
$PY -u -m ml compare --name stage10_deep \
    --run "random start/seed 0=ml/runs/deep_scratch" \
    --run "imitation start/seed 0=ml/runs/deep_from_imitation" \
    --rounds 0 50 100 200 300 400 500 \
    --panel strategic_ai ml/runs/overnight_scratch/round_0500.pt \
    --games 100 || exit 1

echo "=== move scoring on the deep model"
ATLAS_OPPONENT_CHECKPOINT=$DEEP $PY -u -m ml evaluate --candidate "value:$DEEP?candidates=8" \
    --opponents aggressive_ai careful_ai strategic_ai "$DEEP" ml/runs/overnight_scratch/round_0500.pt atlas_clean_ai atlas_ai \
    --generated-maps 20 --no-bundled-maps --skip-generated 30 || exit 1
echo "=== after deep finished"
