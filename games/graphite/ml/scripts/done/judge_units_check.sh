#!/bin/sh
# Unit counts on a log scale or a plain linear one: which lets the judge judge positions better?
#
# Two judges on the same data (round1) with the same settings (weight decay, early stopping), both
# from random weights, differing only in how they read unit counts (units, incoming fleets, units
# once fleets land, a capture's cost): as the encoder gives them (log) or turned back into plain
# counts (linear). Compared by held-out loss, test score error, and calibration.
#
# Run:  nohup ml/scripts/done/judge_units_check.sh > ml/runs/logs/judge_units_check_log.txt 2>&1 &
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
for scale in log linear; do
    [ -f ml/runs/judge_$scale/test.json ] && continue
    echo "=== judge_$scale"
    $PY -u -m ml judge-train --data round1 --run judge_$scale --set unit_scale=$scale --overwrite || exit 1
done
echo "=== judge units check finished"
