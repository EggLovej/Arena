#!/bin/sh
# The front runner trains on under a limit of one send a second (2026-10-05).
#
# Arm C of the ablation in history.md, "Acting once a second": shaping_entropy round 150 continues
# for 150 rounds of self-play in which the network may send at most once every 10 ticks (it sends
# about 7 times a second without the limit). Every other setting is exactly entropy_decay's, which
# is therefore the control: the same start and settings without the limit, every round kept.
# Only the networks of the run are limited; the scripted AIs in the evaluations play as always.
# Smaller gaps (5, 3, 2, 1) are to follow later as their own runs, from this run's last round.
#
# Run:    nohup ml/scripts/done/send_gap.sh >> ml/runs/logs/send_gap_log.txt 2>&1 < /dev/null &
# Watch:  .venv/bin/python -m ml monitor --run send_gap10
# Stop:   pkill -f send_gap.sh; pkill -f "ml train"
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
START=ml/runs/shaping_entropy/round_0150.pt
RUN=send_gap10
ROUNDS=150
trap 'cp ml/runs/deep_from_imitation/round_0500.pt ml/runs/latest.pt' EXIT

if [ ! -f ml/runs/$RUN/latest.pt ]; then
    echo "=== $RUN ($(date +%H:%M))"
    $PY -u -m ml train selfplay --run $RUN --init $START --seed 1 --rounds $ROUNDS --min-send-gap 10 \
        --learning-rate 4.4e-5 --value-only-rounds 0 --material-hint 0 --evaluate-every 10 --final-evaluation no \
        --entropy-weight 0.001 --entropy-half-life 200 --entropy-floor 0.0002 --stop-after 50 --overwrite || exit 1
fi
until grep -q "\"round\": $ROUNDS," ml/runs/$RUN/metrics.jsonl; do
    echo "=== $RUN (resuming, $(date +%H:%M))"
    $PY -u -m ml train selfplay --run $RUN --resume --stop-after 50 || exit 1
done
echo "=== $RUN finished ($(date +%H:%M))"
