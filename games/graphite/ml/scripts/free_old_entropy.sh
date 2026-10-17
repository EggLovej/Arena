#!/bin/sh
# Stage 4, the entropy bonus (2026-10-10, the user's plan): the new design with the old bonus.
#
# free_scratch (v2 inputs, free amounts, entropy bonus 0.001 halving every 200 rounds towards
# 0.0002) beat deep_scratch (v1 inputs, options, bonus 0.01 throughout) at round 300: 1289 against
# 1193, 63% head to head. Both the design and the bonus differed, so this run keeps the design and
# takes the old bonus: free_old_entropy is free_scratch exactly (the same start, seed 0 so the same
# training maps, the same settings) except entropy_weight 0.01, constant. 300 rounds.
#
# Then ml compare: rounds 100, 200 and 300 of all three runs on one scale against strategic_ai,
# aggressive_ai and careful_ai, 100 games each, and the three rounds 300 head to head.
# Expected time: about 4.5 to 6 hours of training (the larger bonus may keep games longer), then
# about 10 minutes for the comparison.
# Resumable: run it again and it carries on where it stopped.
# ml/runs/latest.pt (the game's Neural AI) is set back to deep_from_imitation round 500 at the end.
#
# Run:    nohup ml/scripts/free_old_entropy.sh >> ml/runs/logs/free_old_entropy_log.txt 2>&1 < /dev/null &
# Watch:  .venv/bin/python -m ml monitor --run free_old_entropy
# Stop:   pkill -f free_old_entropy.sh; pkill -f "ml train selfplay --run free_old_entropy"
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
ROUNDS=300
RUN=free_old_entropy
trap 'cp ml/runs/deep_from_imitation/round_0500.pt ml/runs/latest.pt' EXIT

START=ml/runs/deep_starts/free_v2_seed0.pt
[ -f $START ] || $PY -m ml checkpoint new $START --seed 0 --set layers=6 amounts=free feature_set=v2 || exit 1

if [ ! -f ml/runs/$RUN/latest.pt ]; then
    echo "=== $RUN ($(date +%H:%M))"
    $PY -u -m ml train selfplay --run $RUN --init $START --seed 0 --rounds $ROUNDS --evaluate-every 10 \
        --entropy-weight 0.01 --final-evaluation no --overwrite || exit 1
fi
until grep -q "\"round\": $ROUNDS," ml/runs/$RUN/metrics.jsonl; do
    echo "=== $RUN (resuming, $(date +%H:%M))"
    $PY -u -m ml train selfplay --run $RUN --resume || exit 1
done
echo "=== $RUN finished ($(date +%H:%M))"

if [ ! -f ml/runs/compare_free_old_entropy/report.txt ]; then
    echo "=== comparison with free_scratch and deep_scratch ($(date +%H:%M))"
    $PY -u -m ml compare --name free_old_entropy \
        --run "free amounts, v2, bonus 0.01/seed 0=ml/runs/$RUN" \
        --run "free amounts, v2, bonus 0.001 halving/seed 0=ml/runs/free_scratch" \
        --run "options, v1, bonus 0.01/seed 0=ml/runs/deep_scratch" \
        --rounds 100 200 300 --panel strategic_ai aggressive_ai careful_ai --games 100 || exit 1
fi
echo "=== free_old_entropy finished ($(date +%H:%M))"
