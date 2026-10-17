#!/bin/sh
# Move scoring (value:PATH) against the plain network, strategic_ai, and search, all on overnight round 500.
#
# Two versions: judging every legal move, and judging only the policy's 8 favourite sends.
# 40 fresh maps from both seats, so 80 games per pairing; about 25 minutes.
#
# Run:  ml/scripts/done/value_player_check.sh   (results are printed and saved next to this run's log)
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
NET=ml/runs/overnight_scratch/round_0500.pt
for candidate in "value:$NET" "value:$NET?candidates=8"; do
    echo "=== $candidate"
    $PY -u -m ml evaluate --candidate "$candidate" --opponents "$NET" strategic_ai "search:$NET" \
        --generated-maps 40 --no-bundled-maps --skip-generated 30 || exit 1
done
echo "=== value player check finished"
