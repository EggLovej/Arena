#!/bin/sh
# Input check: do the new inputs let the network copy strategic_ai's sends better?
#
# strategic_ai is deterministic and keeps no memory, so a network that sees what it needs could copy
# it closely; eighths cap that at about 93% of its sends. In the capacity probe, no network size got
# past 67-72%. Here the size stays fixed (64 wide, 6 layers, the "deeper" variant at 71%) and only
# what the network reads changes:
#
#   before            the inputs as they were (route_features=0, global_features=4)
#   routes            + what a capture along each route costs now (ROUTE_FEATURES)
#   routes_production + everyone's production as a share of all production (last 3 GLOBAL_FEATURES)
#
# Each is trained twice, on the teacher games of seed 0 and of seed 1 (different maps), for 20 epochs
# with the capacity probe's settings. "before" on seed 0 repeats capacity_deeper and should land on
# its numbers; the two seeds of each show how much a result moves by chance.
#
# Waits for a running look-ahead evaluation to finish first, so the two do not compete for cores.
# Expected time: about 1 h 30 min. Safe to start again: finished runs are skipped.
# ml/runs/latest.pt (what the game's Neural AI plays) is put back as it was when the script ends.
#
# Run:    nohup ml/scripts/done/input_check.sh >> ml/runs/logs/input_check_log.txt 2>&1 < /dev/null &
# Watch:  .venv/bin/python -m ml monitor --run inputs_routes_seed0   (or any run)
# Stop:   pkill -f input_check.sh; pkill -f "ml train imitation"
# Result: ml/runs/logs/input_check_report.txt
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
EPOCHS=20

while pgrep -f "candidate lookahead:" > /dev/null; do sleep 30; done

backup=$(mktemp)
if [ -f ml/runs/latest.pt ]; then cp ml/runs/latest.pt "$backup"; fi
trap 'if [ -s "$backup" ]; then cp "$backup" ml/runs/latest.pt; fi; rm -f "$backup"' EXIT

variant() {  # name, seed, architecture settings...
    run=inputs_$1_seed$2
    seed=$2
    shift 2
    if grep -q "\"round\": $EPOCHS," ml/runs/$run/metrics.jsonl 2>/dev/null; then
        echo "=== $run already finished"
        return 0
    fi
    echo "=== $run: $* ($(date +%H:%M))"
    mkdir -p ml/runs/input_starts
    start=ml/runs/input_starts/$run.pt
    $PY -m ml checkpoint new $start --seed $seed --set "$@" || return 1
    $PY -u -m ml train imitation --run $run --init $start --seed $seed --epochs $EPOCHS \
        --evaluation-maps 0 --final-evaluation no --overwrite || return 1
}

for seed in 0 1; do
    variant before $seed layers=6 route_features=0 global_features=4 || exit 1
    variant routes $seed layers=6 global_features=4 || exit 1
    variant routes_production $seed layers=6 || exit 1
done

$PY -m ml summary capacity_deeper inputs_before_seed0 inputs_before_seed1 inputs_routes_seed0 inputs_routes_seed1 \
    inputs_routes_production_seed0 inputs_routes_production_seed1 \
    > ml/runs/logs/input_check_report.txt || exit 1
cat ml/runs/logs/input_check_report.txt
echo "=== input check finished ($(date +%H:%M))"
