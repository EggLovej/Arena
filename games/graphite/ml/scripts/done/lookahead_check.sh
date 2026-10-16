#!/bin/sh
# Two-move look-ahead against one-move scoring, all on the strongest network (deep_from_imitation round 500).
#
# Stage 1: each of the three rules for choosing from the table (nash, expected, worst), 8 of our moves
# against 8 of theirs, plays one-move scoring (value:..., 8 candidates) and the plain network. 30 fresh
# maps from both seats, 60 games per pairing. About 30 minutes.
# Stage 2 (pass a rule as the first argument): that rule against strategic_ai and the original Atlas
# (its opponent network pinned to the same network). About 30 minutes.
#
# Run:  ml/scripts/done/lookahead_check.sh          stage 1
#       ml/scripts/done/lookahead_check.sh nash     stage 2 with the chosen rule
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
NET=ml/runs/deep_from_imitation/round_0500.pt
MAPS="--generated-maps 30 --no-bundled-maps --skip-generated 30 --save-losses-against"
if [ -z "$1" ]; then
    for rule in nash expected worst; do
        echo "=== lookahead, rule $rule"
        $PY -u -m ml evaluate --candidate "lookahead:$NET?rule=$rule" --opponents "value:$NET?candidates=8" "$NET" $MAPS || exit 1
    done
else
    echo "=== lookahead, rule $1, against strategic_ai and Atlas"
    ATLAS_OPPONENT_CHECKPOINT=$NET $PY -u -m ml evaluate --candidate "lookahead:$NET?rule=$1" \
        --opponents strategic_ai atlas_ai $MAPS || exit 1
fi
echo "=== lookahead check finished"
