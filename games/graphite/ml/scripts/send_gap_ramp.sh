#!/bin/sh
# Lifting the limit step by step (2026-10-06, the user's plan): from send_gap10 round 200, 50 rounds
# each with at most one send every 9, 8, 7, 6 and 5 ticks; then look again.
#
# Each step is its own run (send_gap9 ... send_gap5), started from the last round of the step
# before, so every tool knows which limit a network trained under. The weights carry over; the
# optimizer's memory starts afresh each step. The step size starts where the step before ended,
# and the entropy bonus carries on its schedule (0.001 halving every 200 rounds towards 0.0002,
# counted from send_gap10's round 1). Each step draws its own training maps (seed = its gap).
#
# After each step, ml compare --unlimited rates its last round with the limit lifted, on the
# usual panel plus the control's round 300 and send_gap10 round 200 (where this starts).
# Expected time: about 2 to 2.5 hours, plus about 4 minutes per comparison. Resumable.
# ml/runs/latest.pt (the game's Neural AI) is set back to deep_from_imitation round 500 at the end.
#
# Run:    nohup ml/scripts/send_gap_ramp.sh >> ml/runs/logs/send_gap_ramp_log.txt 2>&1 < /dev/null &
# Watch:  .venv/bin/python -m ml monitor --run send_gap9   (then send_gap8, ...)
# Stop:   pkill -f send_gap_ramp.sh; pkill -f "ml train"
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
ROUNDS=50
PANEL="strategic_ai aggressive_ai ml/runs/deep_from_imitation/round_0400.pt ml/runs/deep_scratch/round_0300.pt
       ml/runs/entropy_decay/round_0300.pt ml/runs/send_gap10/round_0200.pt"
trap 'cp ml/runs/deep_from_imitation/round_0500.pt ml/runs/latest.pt' EXIT

START=ml/runs/send_gap10/round_0200.pt
PREVIOUS_METRICS=ml/runs/send_gap10/metrics.jsonl
PREVIOUS_ROUND=200
BEHIND=200  # rounds of the entropy schedule already behind us
for GAP in 9 8 7 6 5; do
    RUN=send_gap$GAP
    if ! grep -q "\"round\": $ROUNDS," ml/runs/$RUN/metrics.jsonl 2>/dev/null; then
        if [ -f ml/runs/$RUN/latest.pt ]; then
            echo "=== $RUN (resuming, $(date +%H:%M))"
            $PY -u -m ml train selfplay --run $RUN --resume || exit 1
        else
            # The step size where the step before stopped, and the entropy bonus where its schedule is now.
            STEP=$($PY -c "import json; rows = [json.loads(line) for line in open('$PREVIOUS_METRICS')]; print(next(row['learning rate'] for row in rows if row['round'] == $PREVIOUS_ROUND))") || exit 1
            ENTROPY=$($PY -c "print(0.0002 + 0.0008 * 0.5 ** ($BEHIND / 200))") || exit 1
            echo "=== $RUN from $START: step size $STEP, entropy bonus $ENTROPY ($(date +%H:%M))"
            $PY -u -m ml train selfplay --run $RUN --init $START --seed $GAP --rounds $ROUNDS --min-send-gap $GAP \
                --learning-rate $STEP --value-only-rounds 0 --material-hint 0 --evaluate-every 10 --final-evaluation no \
                --entropy-weight $ENTROPY --entropy-half-life 200 --entropy-floor 0.0002 --overwrite || exit 1
        fi
    fi
    if [ ! -f ml/runs/compare_ramp_gap$GAP/results.json ]; then
        echo "=== $RUN with the limit lifted ($(date +%H:%M))"
        $PY -u -m ml compare --name ramp_gap$GAP --unlimited --run "gap $GAP/seed $GAP=ml/runs/$RUN" --rounds $ROUNDS \
            --games 100 --panel $PANEL || exit 1
    fi
    START=$(printf "ml/runs/%s/round_%04d.pt" $RUN $ROUNDS)
    PREVIOUS_METRICS=ml/runs/$RUN/metrics.jsonl
    PREVIOUS_ROUND=$ROUNDS
    BEHIND=$((BEHIND + ROUNDS))
done
echo "=== send_gap_ramp finished ($(date +%H:%M))"
