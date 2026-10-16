#!/bin/sh
# Overnight: self-play only, from random weights, 500 rounds, current default settings (the tuning found nothing better).
# Each process trains at most 50 rounds, then the run carries on in a fresh one (--resume), so the slow memory
# growth of a single process can never build up.
#
# Watch it:   .venv/bin/python -m ml monitor --run overnight_scratch
# Stop it:    pkill -f overnight_scratch.sh; pkill -f "ml train selfplay --run overnight_scratch"
# Carry on:   ml/scripts/done/overnight_scratch.sh   (picks up from the last finished round)
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
RUN=overnight_scratch
if [ ! -f ml/runs/$RUN/metrics.jsonl ]; then
    $PY -u -m ml train selfplay --run $RUN --init random --rounds 500 --evaluate-every 10 --stop-after 50 || exit 1
fi
until grep -q '"round": 500,' ml/runs/$RUN/metrics.jsonl; do
    $PY -u -m ml train selfplay --run $RUN --resume --stop-after 50 || exit 1
done
echo "=== overnight run finished"
