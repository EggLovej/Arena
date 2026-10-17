#!/bin/sh
# A third judge for Beast, from a stronger player's games (2026-10-04).
#
# judge1 and judge2 learned from games of deep_from_imitation round 500; four times the games
# (judge2) played no better. Round 3 comes from today's best network instead (entropy_decay round
# 300, about +130 linked rating over round 500), so the judge learns how games go between
# stronger players.
#
# 1. Games (ml judge-games, round3, seed 2): 30,000 against itself, 1,500 each against
#    aggressive_ai, careful_ai, strategic_ai, deep_from_imitation round 500 (the old player) and
#    deep_scratch round 500 (the 64x6 trained without imitation); 8 positions per seat.
# 2. Two judges, linear unit counts, from random weights: judge3 on round3 only, judge3_all on
#    rounds 1-3 together.
# 3. The better of the two in play: their look-aheads (the new player proposing the moves) head to
#    head, 100 games.
# 4. The winner's full check (ml judge-check), with the new player proposing: against the old judge
#    (the network's own value output), Beast, Atlas (forecasting with round 500), the plain
#    network, and the saved losses.
# 5. The judge alone: the look-ahead with Beast's own proposer (round 500) and the new judge,
#    against Beast, 100 games.
#
# Expected: about 1.2 h of games, 1 h of training, 1.5 h of checks.
# Safe to start again: finished steps are skipped (the games carry on where they stopped).
# ml/runs/latest.pt (the game's Neural AI) is set back to deep_from_imitation round 500 at the end.
#
# Run:    nohup ml/scripts/judge3.sh >> ml/runs/logs/judge3_log.txt 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
PLAYER=ml/runs/entropy_decay/round_0300.pt
OLD=ml/runs/deep_from_imitation/round_0500.pt
BEAST="lookahead:$OLD?rule=nash&judge=ml/runs/judge1/best.pt"
MAPS="--generated-maps 50 --no-bundled-maps --skip-generated 30 --workers 11"
trap 'cp ml/runs/deep_from_imitation/round_0500.pt ml/runs/latest.pt' EXIT

echo "=== 1. games ($(date +%H:%M))"
if [ -d ml/runs/judge_data/round3 ]; then
    $PY -u -m ml judge-games --name round3 --resume || exit 1
else
    $PY -u -m ml judge-games --name round3 --player $PLAYER --self-games 30000 --games-per-opponent 1500 \
        --opponents aggressive_ai careful_ai strategic_ai $OLD ml/runs/deep_scratch/round_0500.pt \
        --positions-per-seat 8 --seed 2 --workers 11 || exit 1
fi

echo "=== 2. judges ($(date +%H:%M))"
if [ ! -f ml/runs/judge3/test.json ]; then  # written when training has finished
    $PY -u -m ml judge-train --data round3 --run judge3 --set unit_scale=linear --compare-with $PLAYER --overwrite || exit 1
fi
if [ ! -f ml/runs/judge3_all/test.json ]; then
    $PY -u -m ml judge-train --data round1 round2 round3 --run judge3_all --set unit_scale=linear --compare-with $PLAYER --overwrite || exit 1
fi

echo "=== 3. judge3 against judge3_all ($(date +%H:%M))"
if [ ! -f ml/runs/logs/judge3_pick.txt ]; then
    $PY -u -m ml evaluate --candidate "lookahead:$PLAYER?rule=nash&judge=ml/runs/judge3/best.pt" \
        --opponents "lookahead:$PLAYER?rule=nash&judge=ml/runs/judge3_all/best.pt" $MAPS > ml/runs/logs/judge3_pick.part || exit 1
    mv ml/runs/logs/judge3_pick.part ml/runs/logs/judge3_pick.txt
fi
grep -E "^ *(games|vs |overall)" ml/runs/logs/judge3_pick.txt
SCORE=$(grep "^overall" ml/runs/logs/judge3_pick.txt | awk '{print $6}' | tr -d '%')
if awk "BEGIN {exit !($SCORE >= 50)}"; then JUDGE=ml/runs/judge3/best.pt; else JUDGE=ml/runs/judge3_all/best.pt; fi
echo "=== the better judge: $JUDGE (judge3 scored $SCORE%)"

echo "=== 4. full check ($(date +%H:%M))"
$PY -u -m ml judge-check --judge $JUDGE --network $PLAYER --atlas-network $OLD --workers 11 || exit 1

echo "=== 5. the judge alone: Beast's proposer with the new judge, against Beast ($(date +%H:%M))"
if [ ! -f ml/runs/logs/judge3_alone.txt ]; then
    $PY -u -m ml evaluate --candidate "lookahead:$OLD?rule=nash&judge=$JUDGE" --opponents "$BEAST" $MAPS \
        > ml/runs/logs/judge3_alone.part || exit 1
    mv ml/runs/logs/judge3_alone.part ml/runs/logs/judge3_alone.txt
fi
grep -E "^ *(games|vs |overall)" ml/runs/logs/judge3_alone.txt
echo "=== judge3 finished ($(date +%H:%M))"
