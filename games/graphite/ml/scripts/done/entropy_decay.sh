#!/bin/sh
# The front runner trains on, with a shrinking reward for staying undecided (2026-10-04).
#
# shaping_entropy round 150 (round 1050 of its lineage: imitation, deep_from_imitation 500 rounds,
# ab_plain 400, shaping_entropy 150) continues for 300 rounds. Its entropy bonus starts at 0.001
# (where it is now) and halves every 200 rounds towards a floor of 0.0002, never below. Other
# settings as before: step size 4.4e-5 to start, no value-only rounds, no material hint.
#
# Then ml compare of rounds 0, 100, 200, 300 on the usual panel, and the lineage view updated.
# Expected time: about 3.5 hours of training, 10 minutes of comparison.
# ml/runs/latest.pt (the game's Neural AI) is set back to deep_from_imitation round 500 at the end.
#
# Run:    nohup ml/scripts/done/entropy_decay.sh >> ml/runs/logs/entropy_decay_log.txt 2>&1 &
# Watch:  .venv/bin/python -m ml monitor --run entropy_decay
# Stop:   pkill -f entropy_decay.sh; pkill -f "ml train"
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
START=ml/runs/shaping_entropy/round_0150.pt
RUN=entropy_decay
ROUNDS=300
trap 'cp ml/runs/deep_from_imitation/round_0500.pt ml/runs/latest.pt' EXIT

if [ ! -f ml/runs/$RUN/latest.pt ]; then
    echo "=== $RUN ($(date +%H:%M))"
    $PY -u -m ml train selfplay --run $RUN --init $START --seed 1 --rounds $ROUNDS \
        --learning-rate 4.4e-5 --value-only-rounds 0 --material-hint 0 --evaluate-every 10 --final-evaluation no \
        --entropy-weight 0.001 --entropy-half-life 200 --entropy-floor 0.0002 --stop-after 50 --overwrite || exit 1
fi
until grep -q "\"round\": $ROUNDS," ml/runs/$RUN/metrics.jsonl; do
    echo "=== $RUN (resuming, $(date +%H:%M))"
    $PY -u -m ml train selfplay --run $RUN --resume --stop-after 50 || exit 1
done
echo "=== $RUN finished ($(date +%H:%M))"

echo "=== comparison ($(date +%H:%M))"
$PY -u -m ml compare --name entropy_decay --run "decay/seed 1=ml/runs/$RUN" --rounds 0 100 200 300 \
    --panel strategic_ai aggressive_ai ml/runs/deep_from_imitation/round_0400.pt ml/runs/deep_scratch/round_0300.pt \
    --games 100 || exit 1
$PY -m ml lineage ml/runs/$RUN/round_0300.pt
echo "=== entropy decay finished ($(date +%H:%M))"
