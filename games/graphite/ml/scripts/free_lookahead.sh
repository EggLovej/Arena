#!/bin/sh
# A longer lookahead (2026-10-10, the user's plan): free_scratch again, with lookahead 0.995.
#
# The lookahead says how far ahead a move is credited with what follows: at 0.99 (every run so far)
# half its credit is gone after 6.3 s of game time, at 0.995 after 11.5 s. Even playing their
# favourite, our networks send one in five sends between own bases away from the front and a third
# straight back; such a shuffle only costs something seconds later. free_lookahead is free_scratch
# exactly (the same start free_v2_seed0.pt, seed 0 so the same training maps, the same settings and
# entropy schedule: 0.001 halving every 200 rounds towards 0.0002), except --lookahead 0.995.
# 300 rounds, then ml compare: rounds 100, 200 and 300 of both on one scale against strategic_ai,
# aggressive_ai and careful_ai, 100 games each, and both rounds 300 head to head.
# Expected time: about 4.5 to 5 hours of training, then about 10 minutes for the comparison.
# Resumable: run it again and it carries on where it stopped.
# ml/runs/latest.pt (the game's Neural AI) is set back to deep_from_imitation round 500 at the end.
#
# Run:    nohup ml/scripts/free_lookahead.sh >> ml/runs/logs/free_lookahead_log.txt 2>&1 < /dev/null &
# Watch:  .venv/bin/python -m ml tensorboard --run free_scratch --run free_lookahead
# Stop:   pkill -f free_lookahead.sh; pkill -f "ml train selfplay --run free_lookahead"
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
ROUNDS=300
RUN=free_lookahead
trap 'cp ml/runs/deep_from_imitation/round_0500.pt ml/runs/latest.pt' EXIT

START=ml/runs/deep_starts/free_v2_seed0.pt
[ -f $START ] || $PY -m ml checkpoint new $START --seed 0 --set layers=6 amounts=free feature_set=v2 || exit 1

if [ ! -f ml/runs/$RUN/latest.pt ]; then
    echo "=== $RUN ($(date +%H:%M))"
    $PY -u -m ml train selfplay --run $RUN --init $START --seed 0 --rounds $ROUNDS --evaluate-every 10 \
        --entropy-weight 0.001 --entropy-half-life 200 --entropy-floor 0.0002 --lookahead 0.995 \
        --final-evaluation no --overwrite || exit 1
fi
until grep -q "\"round\": $ROUNDS," ml/runs/$RUN/metrics.jsonl; do
    echo "=== $RUN (resuming, $(date +%H:%M))"
    $PY -u -m ml train selfplay --run $RUN --resume || exit 1
done
echo "=== $RUN finished ($(date +%H:%M))"

if [ ! -f ml/runs/compare_free_lookahead/report.txt ]; then
    echo "=== comparison with free_scratch ($(date +%H:%M))"
    $PY -u -m ml compare --name free_lookahead \
        --run "lookahead 0.995/seed 0=ml/runs/$RUN" \
        --run "lookahead 0.99/seed 0=ml/runs/free_scratch" \
        --rounds 100 200 300 --panel strategic_ai aggressive_ai careful_ai --games 100 || exit 1
fi
echo "=== free_lookahead finished ($(date +%H:%M))"
