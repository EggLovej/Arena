#!/bin/sh
# Stage 4, first look (2026-10-09, the user's plan): the v2 inputs with free amounts, self-play only
# from random weights, 100 rounds; then rated next to deep_scratch, the old design trained the same way.
#
# 1. free_scratch: a fresh 64x6 network (as deep_scratch) with feature_set=v2 and amounts=free.
#    Settings as deep_scratch (seed 0, so the same training maps; 96 games a round; 2 value-only
#    rounds; the material hint fading over 50 rounds; evaluation every 10 rounds), except the
#    entropy bonus: 0.001 halving every 200 rounds towards 0.0002 (deep_scratch: 0.01 throughout,
#    which later made the amounts nearly random; free amounts feel the bonus more). No send limit.
# 2. ml compare: rounds 0, 50 and 100 of both runs on one scale against strategic_ai,
#    aggressive_ai and careful_ai, 100 games each, and both rounds 100 head to head.
#
# Each training process stops after 50 rounds and the next one resumes, as in deep_scratch's run.
# Expected time: about 2.5 to 3 hours of training, then about 20 minutes for the comparison.
# Resumable: run it again and it carries on where it stopped.
# ml/runs/latest.pt (the game's Neural AI) is set back to deep_from_imitation round 500 at the end.
#
# Run:    nohup ml/scripts/free_scratch.sh >> ml/runs/logs/free_scratch_log.txt 2>&1 < /dev/null &
# Watch:  .venv/bin/python -m ml monitor --run free_scratch
# Stop:   pkill -f free_scratch.sh; pkill -f "ml train selfplay --run free_scratch"
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
ROUNDS=100
RUN=free_scratch
trap 'cp ml/runs/deep_from_imitation/round_0500.pt ml/runs/latest.pt' EXIT

START=ml/runs/deep_starts/free_v2_seed0.pt
[ -f $START ] || $PY -m ml checkpoint new $START --seed 0 --set layers=6 amounts=free feature_set=v2 || exit 1

if [ ! -f ml/runs/$RUN/latest.pt ]; then
    echo "=== $RUN ($(date +%H:%M))"
    $PY -u -m ml train selfplay --run $RUN --init $START --seed 0 --rounds $ROUNDS --evaluate-every 10 \
        --entropy-weight 0.001 --entropy-half-life 200 --entropy-floor 0.0002 --final-evaluation no \
        --stop-after 50 --overwrite || exit 1
fi
until grep -q "\"round\": $ROUNDS," ml/runs/$RUN/metrics.jsonl; do
    echo "=== $RUN (resuming, $(date +%H:%M))"
    $PY -u -m ml train selfplay --run $RUN --resume --stop-after 50 || exit 1
done

if [ ! -f ml/runs/compare_free_scratch/report.txt ]; then
    echo "=== comparison with deep_scratch ($(date +%H:%M))"
    $PY -u -m ml compare --name free_scratch \
        --run "free amounts, v2/seed 0=ml/runs/$RUN" \
        --run "options, v1/seed 0=ml/runs/deep_scratch" \
        --rounds 0 50 100 --panel strategic_ai aggressive_ai careful_ai --games 100 || exit 1
fi
echo "=== free_scratch finished ($(date +%H:%M))"
