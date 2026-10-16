#!/bin/sh
# Everything after round 2 of the judge's games, unattended:
#
# 1. Train judge2 on round1 + round2 (linear unit counts, random start, weight decay, early stopping).
# 2. Test it in play (ml judge-check): against the old judge, against Beast (the look-ahead with
#    judge1), against Atlas and the plain network, and on the saved lost starts.
# 3. The better judge (judge2 if it scores at least 50% against Beast, else judge1) teaches the policy:
#    policy-targets (teach1), policy-train (policy1), policy-check.
#
# Waits for round 2 to finish first. Safe to start again: finished steps are skipped.
# Beast is not changed here; that is decided from the results.
#
# Run:  nohup ml/scripts/done/after_round2.sh > ml/runs/logs/after_round2_log.txt 2>&1 < /dev/null &
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python

echo "=== waiting for round 2 ($(date +%H:%M))"
while pgrep -f "[j]udge-games --name round2" > /dev/null; do sleep 30; done
$PY -c "
import json, sys
s = json.load(open('ml/runs/judge_data/round2/progress.json'))
targets = {'itself': s['settings']['self_games'], **{o: s['settings']['games_per_opponent'] for o in s['settings']['opponents']}}
sys.exit(0 if all(s['games'][g] >= n for g, n in targets.items()) else 1)
" || { echo "round 2 stopped before it was complete; not going on"; exit 1; }
echo "=== step 0 done: round 2 complete ($(date +%H:%M))"

if [ ! -f ml/runs/judge2/test.json ]; then
    $PY -u -m ml judge-train --data round1 round2 --run judge2 --set unit_scale=linear --overwrite || exit 1
fi
echo "=== step 1 done: judge2 trained ($(date +%H:%M))"

if [ ! -f ml/runs/judge2/check.json ]; then
    $PY -u -m ml judge-check --judge ml/runs/judge2/best.pt || exit 1
fi
echo "=== step 2 done: judge2 tested ($(date +%H:%M))"

BEST=$($PY -c "
import json
c = json.load(open('ml/runs/judge2/check.json'))
s = c.get('vs Beast')
score = (s['wins'] + s['draws'] / 2) / s['games'] if s and s['games'] else 0.0
print('ml/runs/judge2/best.pt' if score >= 0.5 else 'ml/runs/judge1/best.pt')
")
echo "the policy is taught with: $BEST"
echo "$BEST" > ml/runs/logs/after_round2_teaching_judge.txt

if [ ! -f ml/runs/judge_data/teach1/progress.json ]; then
    $PY -u -m ml policy-targets --name teach1 --judge "$BEST" --self-games 2500 --games-per-opponent 100 || exit 1
fi
if [ ! -f ml/runs/policy1/summary.json ]; then
    $PY -u -m ml policy-train --targets teach1 --run policy1 --overwrite || exit 1
fi
echo "=== step 3a done: policy1 taught ($(date +%H:%M))"
if [ ! -f ml/runs/policy1/check.json ]; then
    $PY -u -m ml policy-check --policy ml/runs/policy1/best.pt --judge "$BEST" || exit 1
fi
echo "=== step 3 done: policy1 tested ($(date +%H:%M))"
echo "=== after round 2 finished"
