#!/bin/sh
# The run under a limit of one send a second trains on, rounds 151-300 (2026-10-06).
#
# send_gap10 stopped at round 150, where the network, with the limit lifted, beat the unlimited
# control 61% (and the control's round 300 60.5%), more so the longer it had trained under the
# limit (history.md, "Acting once a second"). This carries the same run on to round 300, exactly
# where it stopped (weights, optimizer, step size, entropy schedule), to see whether that keeps
# rising. A resumed run keeps the settings it was started with, so its round count is raised in
# its settings.json first; nothing else changes.
#
# Then ml compare --unlimited: rounds 150-300 with the limit lifted, on the usual panel plus the
# control's round 300 (entropy_decay, unlimited, also 300 rounds).
# Expected time: about 55 minutes of training, 10-20 minutes of comparison.
# ml/runs/latest.pt (the game's Neural AI) is set back to deep_from_imitation round 500 at the end.
#
# Run:    nohup ml/scripts/send_gap_more.sh >> ml/runs/logs/send_gap_more_log.txt 2>&1 < /dev/null &
# Watch:  .venv/bin/python -m ml monitor --run send_gap10
# Stop:   pkill -f send_gap_more.sh; pkill -f "ml train"
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
RUN=send_gap10
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

echo "=== comparison with the limit lifted ($(date +%H:%M))"
$PY -u -m ml compare --name send_gap10_unleashed --unlimited --run "unleashed/seed 1=ml/runs/$RUN" \
    --rounds 150 200 250 300 --games 100 \
    --panel strategic_ai aggressive_ai ml/runs/deep_from_imitation/round_0400.pt ml/runs/deep_scratch/round_0300.pt \
    ml/runs/entropy_decay/round_0300.pt || exit 1
echo "=== send_gap_more finished ($(date +%H:%M))"
