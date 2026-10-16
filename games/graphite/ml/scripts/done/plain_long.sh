#!/bin/sh
# Plain self-play carried on: ab_plain (from deep_from_imitation round 500) from round 150 to 600.
#
# The night of 2026-10-03 showed plain self-play is still the only thing that keeps improving the
# network (+50 rating points in 150 rounds); the judge as critic, frozen or learning, made it worse.
# ab_plain's settings.json was changed from 150 to 600 rounds; everything else carries on as it was.
# About 7 minutes per 10 rounds, so roughly 5.5 hours. Stop at any time; the rounds so far are kept.
#
# Run:    nohup ml/scripts/done/plain_long.sh > ml/runs/logs/plain_long_log.txt 2>&1 &
# Stop:   pkill -f plain_long.sh; pkill -f "ml train"
# ml/runs/latest.pt (the game's Neural AI) follows the run while it trains; put back at the end.
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
trap 'cp ml/runs/deep_from_imitation/round_0500.pt ml/runs/latest.pt' EXIT
until grep -q '"round": 600,' ml/runs/ab_plain/metrics.jsonl; do
    $PY -u -m ml train selfplay --run ab_plain --resume --stop-after 50 || exit 1
done
echo "=== ab_plain reached round 600 ($(date +%H:%M))"
