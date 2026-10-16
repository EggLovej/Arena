#!/bin/sh
# Third arm of the critic A/B: the judge as critic, but one that keeps learning.
#
# With judge2 frozen as the critic (ab_critic), the network learned to steer into positions the
# judge misjudges in its favour: the critic's error tripled and the network fell apart. Here the
# critic starts as judge2 and after every round takes one pass over a sample of that round's
# positions towards how the games really ended (--critic-learning-rate), so a blind spot the
# policy finds gets corrected. Otherwise identical to ab_plain and ab_critic (same start, seed,
# settings), so ab_plain is its control.
#
# Waits until ml/scripts/done/critic_ab.sh has finished its training (its comparison may still run).
# Then compares rounds 0, 50, 100 and 150 with ab_plain's on the same panel.
#
# Expected time: about 2.5 hours of training, 40 minutes of comparison.
# ml/runs/latest.pt is set back to deep_from_imitation round 500 (the game's Neural AI) at the end.
#
# Run:    nohup ml/scripts/done/critic_learning.sh > ml/runs/logs/critic_learning_log.txt 2>&1 &
# Watch:  .venv/bin/python -m ml monitor --run ab_critic_learning
# Result: ml/runs/compare_critic_learning/
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
START=ml/runs/deep_from_imitation/round_0500.pt
ROUNDS=150
RUN=ab_critic_learning
trap 'cp $START ml/runs/latest.pt' EXIT

until grep -q "=== comparison" ml/runs/logs/critic_ab_log.txt; do sleep 30; done

if [ ! -f ml/runs/$RUN/latest.pt ]; then
    echo "=== $RUN"
    $PY -u -m ml train selfplay --run $RUN --init $START --seed 0 --rounds $ROUNDS \
        --learning-rate 4.4e-5 --value-only-rounds 0 --material-hint 0 --evaluate-every 10 --final-evaluation no \
        --critic ml/runs/judge2/best.pt --critic-learning-rate 1e-4 --stop-after 50 --overwrite || exit 1
fi
until grep -q "\"round\": $ROUNDS," ml/runs/$RUN/metrics.jsonl; do
    echo "=== $RUN (resuming)"
    $PY -u -m ml train selfplay --run $RUN --resume --stop-after 50 || exit 1
done
echo "=== $RUN finished ($(date +%H:%M))"

echo "=== comparison"
$PY -u -m ml compare --name critic_learning \
    --run "plain/seed 0=ml/runs/ab_plain" \
    --run "learning critic/seed 0=ml/runs/$RUN" \
    --rounds 0 50 100 150 \
    --panel strategic_ai aggressive_ai ml/runs/deep_from_imitation/round_0400.pt \
    --games 100 || exit 1
echo "=== critic learning finished ($(date +%H:%M))"
