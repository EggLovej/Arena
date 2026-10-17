#!/bin/sh
# Evaluation of ab_plain round 400 (deep_from_imitation round 500 + 400 rounds of plain self-play).
# 1. Ratings of rounds 0, 150, 250, 400 on last night's panel, plus deep_scratch round 300
#    (the deep network trained without imitation) for reference.
# 2. Round 400 as the look-ahead's network (judge2) against Beast and Atlas, and on instinct
#    against deep_from_imitation round 500.
# Both parts show a live board in the terminal: every match's wins, draws and losses so far,
# updated after each game, with the time left. About 25 minutes for part 1 and 55 for part 2.
#
# Run:    ml/scripts/done/plain_400_check.sh
# Result: ml/runs/compare_plain_400/report.txt and ml/runs/ab_plain/check.json
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
echo "=== 1/2: ratings of rounds 0, 150, 250 and 400 ($(date +%H:%M))"
$PY -u -m ml compare --name plain_400 \
    --run "plain/seed 0=ml/runs/ab_plain" \
    --rounds 0 150 250 400 \
    --panel strategic_ai aggressive_ai ml/runs/deep_from_imitation/round_0400.pt ml/runs/deep_scratch/round_0300.pt \
    --games 100 || exit 1
echo "=== 2/2: round 400 with the look-ahead against Beast and Atlas ($(date +%H:%M))"
$PY -u -m ml policy-check --policy ml/runs/ab_plain/round_0400.pt --judge ml/runs/judge2/best.pt --workers 6 || exit 1
echo "=== plain 400 check finished ($(date +%H:%M))"
