#!/bin/sh
# A larger network with tactical amounts and a capture output, warm-started from the best network.
#
# The new setup (2026-10-03): 128 wide x 8 layers (1.1 million weights, five times the current
# 64x6), the amount options "hold" and "surplus" (ml/encoding/action_encoder.py), and an extra
# output that learns, for every send, whether its target was ours once the fleet landed.
#
# 1. A fresh 128x8 network with the capture output.
# 2. Kickstart (ml/training/kickstart.py): it learns to play like ab_plain round 400 (in effect
#    round 900 of deep_from_imitation, the best network so far) from 1,200 of its games: its
#    probability for every move, its value, and where each send ended up. The new options get 5%
#    of each route's amount probability where they are on offer, so self-play can try them.
# 3. How close the copy came: the kickstarted network on instinct against its teacher, 200 games.
# 4. Self-play, 150 rounds each, the same settings in both arms (as ab_plain: step size 4.4e-5,
#    no value-only rounds, no material hint, 96 games per round, networks on the CPU workers):
#       large_selfplay   the kickstarted 128x8 network
#       small_selfplay   the teacher itself, 64x6, as it is (control)
# 5. ml compare: rounds 0, 50, 100, 150 of both arms on fresh maps against strategic_ai,
#    aggressive_ai, deep_from_imitation round 400 and deep_scratch round 300, 100 games each,
#    and the last rounds head to head.
#
# Expected time: about 30 minutes for steps 1-3, 3.3 hours for the large arm (about 80 s per
# round), 2 hours for the control (about 50 s per round), 40 minutes for the comparison.
# Safe to start again: finished steps are skipped. ml/runs/latest.pt is set back to
# deep_from_imitation round 500 (the game's Neural AI) when the script ends.
#
# Run:    nohup ml/scripts/done/large_network.sh > ml/runs/logs/large_network_log.txt 2>&1 &
# Watch:  .venv/bin/python -m ml monitor --run large_selfplay   (or large_kickstart, small_selfplay)
# Stop:   pkill -f large_network.sh; pkill -f "ml train"
# Result: ml/runs/compare_large_network/report.txt
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
TEACHER=ml/runs/ab_plain/round_0400.pt
ROUNDS=150
SHARED="--learning-rate 4.4e-5 --value-only-rounds 0 --material-hint 0 --evaluate-every 10 --final-evaluation no"
trap 'cp ml/runs/deep_from_imitation/round_0500.pt ml/runs/latest.pt' EXIT

if [ ! -f ml/runs/large_start.pt ]; then
    echo "=== 1. a fresh 128x8 network ($(date +%H:%M))"
    $PY -m ml checkpoint new ml/runs/large_start.pt --set width=128 layers=8 capture_head=yes --seed 0 || exit 1
fi
if [ ! -f ml/runs/large_kickstart/best.pt ]; then
    echo "=== 2. kickstart from $TEACHER ($(date +%H:%M))"
    $PY -u -m ml train kickstart --run large_kickstart --init ml/runs/large_start.pt --teacher $TEACHER --overwrite || exit 1
fi
if [ ! -f ml/runs/large_kickstart/instinct.txt ]; then
    echo "=== 3. the kickstarted network on instinct against its teacher ($(date +%H:%M))"
    $PY -u -m ml evaluate --candidate ml/runs/large_kickstart/best.pt --opponents $TEACHER \
        --generated-maps 100 --no-bundled-maps --skip-generated 30 > ml/runs/large_kickstart/instinct.part || exit 1
    mv ml/runs/large_kickstart/instinct.part ml/runs/large_kickstart/instinct.txt
fi
tail -4 ml/runs/large_kickstart/instinct.txt

arm() {  # run name, starting checkpoint
    if [ ! -f ml/runs/$1/latest.pt ]; then
        echo "=== 4. $1 from $2 ($(date +%H:%M))"
        $PY -u -m ml train selfplay --run $1 --init $2 --seed 0 --rounds $ROUNDS $SHARED --stop-after 50 --overwrite || return 1
    fi
    until grep -q "\"round\": $ROUNDS," ml/runs/$1/metrics.jsonl; do
        echo "=== $1 (resuming, $(date +%H:%M))"
        $PY -u -m ml train selfplay --run $1 --resume --stop-after 50 || return 1
    done
    echo "=== $1 finished ($(date +%H:%M))"
}
arm large_selfplay ml/runs/large_kickstart/best.pt || exit 1
arm small_selfplay $TEACHER || exit 1

echo "=== 5. comparison ($(date +%H:%M))"
$PY -u -m ml compare --name large_network \
    --run "large/seed 0=ml/runs/large_selfplay" \
    --run "small/seed 0=ml/runs/small_selfplay" \
    --rounds 0 50 100 150 \
    --panel strategic_ai aggressive_ai ml/runs/deep_from_imitation/round_0400.pt ml/runs/deep_scratch/round_0300.pt \
    --games 100 || exit 1
echo "=== large network finished ($(date +%H:%M))"
