#!/bin/sh
# Stage 4, continued (2026-10-09, the user's plan): free_scratch trains on, rounds 101-300.
#
# free_scratch (v2 inputs, free amounts, self-play from random weights) stopped at round 100 about
# even with deep_scratch (1122 against 1108; 76% against 61.5% vs strategic_ai, 44% head to head).
# deep_scratch already went to round 500, so only this run trains on. This is the same run, carried
# on exactly where it stopped (weights, optimizer, step size, entropy schedule: the bonus goes on
# from 0.00077 at round 100 to 0.00048 at round 300). A resumed run keeps the settings it was
# started with, so its round count is raised in its settings.json first; nothing else changes.
#
# Then ml compare: rounds 100, 200 and 300 of both runs on one scale against strategic_ai,
# aggressive_ai and careful_ai, 100 games each, and both rounds 300 head to head.
# Expected time: about 3.3 hours of training (four processes of 50 rounds), then about 10 minutes.
# Resumable: run it again and it carries on where it stopped.
# ml/runs/latest.pt (the game's Neural AI) is set back to deep_from_imitation round 500 at the end.
#
# Run:    nohup ml/scripts/free_scratch_more.sh >> ml/runs/logs/free_scratch_log.txt 2>&1 < /dev/null &
# Watch:  .venv/bin/python -m ml monitor --run free_scratch
# Stop:   pkill -f free_scratch_more.sh; pkill -f "ml train selfplay --run free_scratch"
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
RUN=free_scratch
ROUNDS=300
trap 'cp ml/runs/deep_from_imitation/round_0500.pt ml/runs/latest.pt' EXIT

$PY -c "
import json, pathlib
path = pathlib.Path('ml/runs/$RUN/settings.json')
settings = json.loads(path.read_text())
settings['rounds'] = $ROUNDS
path.write_text(json.dumps(settings, indent=2) + '\n')
" || exit 1
until grep -q "\"round\": $ROUNDS," ml/runs/$RUN/metrics.jsonl; do
    echo "=== $RUN (resuming, $(date +%H:%M))"
    $PY -u -m ml train selfplay --run $RUN --resume --stop-after 50 || exit 1
done
echo "=== $RUN finished ($(date +%H:%M))"

if [ ! -f ml/runs/compare_free_scratch_300/report.txt ]; then
    echo "=== comparison with deep_scratch to round 300 ($(date +%H:%M))"
    $PY -u -m ml compare --name free_scratch_300 \
        --run "free amounts, v2/seed 0=ml/runs/$RUN" \
        --run "options, v1/seed 0=ml/runs/deep_scratch" \
        --rounds 100 200 300 --panel strategic_ai aggressive_ai careful_ai --games 100 || exit 1
fi
echo "=== free_scratch_more finished ($(date +%H:%M))"
