#!/bin/sh
# Stage 3 demo (2026-10-09): does the whole chain run for a network with the v2 inputs and free
# amounts? A small version of stage 4, not the experiment itself.
#
# 1. A fresh 64x6 network (the teacher's size) with feature_set=v2 and amounts=free.
# 2. Kickstart: it learns to play like our best plain network, send_gap5 round 50 (the end of the
#    ramp; unlimited here, as kickstart's teacher games have no send limit). 96 teacher games,
#    4 epochs, against the default's 1200 and 20.
# 3. Self-play: 5 rounds of 48 games, the evaluation (4 maps) at the end.
# 4. How the three play: ml behaviour on the teacher, the kickstarted network and the last round.
#
# Expected time: about 15 minutes.
# ml/runs/latest.pt (the game's Neural AI) is set back to deep_from_imitation round 500 at the end.
#
# Run:    nohup ml/scripts/free_demo.sh >> ml/runs/logs/free_demo_log.txt 2>&1 < /dev/null &
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
TEACHER=ml/runs/send_gap5/round_0050.pt
trap 'cp ml/runs/deep_from_imitation/round_0500.pt ml/runs/latest.pt' EXIT

echo "=== start ($(date +%H:%M))"
$PY -m ml checkpoint new ml/runs/free_demo_start.pt --seed 0 --set layers=6 amounts=free feature_set=v2 || exit 1

echo "=== kickstart ($(date +%H:%M))"
$PY -u -m ml train kickstart --run free_demo_kickstart --init ml/runs/free_demo_start.pt \
    --teacher $TEACHER --games 96 --epochs 4 --patience 0 || exit 1

echo "=== self-play ($(date +%H:%M))"
$PY -u -m ml train selfplay --run free_demo_selfplay --init ml/runs/free_demo_kickstart/best.pt \
    --rounds 5 --games-per-round 48 --value-only-rounds 0 --material-hint 0 --entropy-weight 0.0004 \
    --evaluation-maps 4 --evaluate-every 5 --final-evaluation no || exit 1

echo "=== behaviour ($(date +%H:%M))"
$PY -u -m ml behaviour $TEACHER ml/runs/free_demo_kickstart/best.pt ml/runs/free_demo_selfplay/latest.pt --games 24 || exit 1
echo "=== done ($(date +%H:%M))"
