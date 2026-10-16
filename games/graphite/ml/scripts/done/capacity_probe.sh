#!/bin/sh
# Capacity probe: is the network too small, and if so, where?
#
# Every variant imitates strategic_ai on exactly the same 600 teacher games (same seed, and the
# scripted AIs never use chance), for 20 epochs with the same settings. strategic_ai is
# deterministic, so a network with enough capacity could copy it almost perfectly. Today's network
# agrees on 99% of its waits but only 64-67% of its sends. A variant that clearly beats that shows
# where capacity is missing; if none does, the limit is the inputs or the data, not the size.
#
#   current     64 wide, 3 layers, 4 heads, 1 route layer    (120k weights)
#   wider       128 wide                                      (470k)
#   deeper      6 layers                                      (221k)
#   route_head  2 route layers                                (124k)
#   bigger      128 wide, 4 layers, 8 heads, 2 route layers   (620k)
#   largest     192 wide, 4 layers, 8 heads, 2 route layers   (1.38M), the upper bound: if even this
#               does not copy the sends better, size is not what limits the network
#
# Expected time: about 1 h 20 min. Safe to start again: finished variants are skipped.
# ml/runs/latest.pt (what the game's Neural AI plays) is put back as it was when the script ends.
#
# Run:    ml/scripts/done/capacity_probe.sh
# Watch:  .venv/bin/python -m ml monitor --run capacity_wider   (or any variant)
# Stop:   pkill -f capacity_probe.sh; pkill -f "ml train imitation"
# Result: ml/runs/logs/capacity_probe_report.txt
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
EPOCHS=20

backup=$(mktemp)
if [ -f ml/runs/latest.pt ]; then cp ml/runs/latest.pt "$backup"; fi
trap 'if [ -s "$backup" ]; then cp "$backup" ml/runs/latest.pt; fi; rm -f "$backup"' EXIT

variant() {  # name, architecture settings...
    run=capacity_$1
    shift
    if grep -q "\"round\": $EPOCHS," ml/runs/$run/metrics.jsonl 2>/dev/null; then
        echo "=== $run already finished"
        return 0
    fi
    echo "=== $run: $*"
    mkdir -p ml/runs/capacity_starts
    start=ml/runs/capacity_starts/$run.pt
    if [ $# -gt 0 ]; then
        $PY -m ml checkpoint new $start --seed 0 --set "$@" || return 1
    else
        $PY -m ml checkpoint new $start --seed 0 || return 1
    fi
    $PY -u -m ml train imitation --run $run --init $start --seed 0 --epochs $EPOCHS \
        --evaluation-maps 0 --final-evaluation no --overwrite || return 1
}

variant current || exit 1
variant wider width=128 || exit 1
variant deeper layers=6 || exit 1
variant route_head route_layers=2 || exit 1
variant bigger width=128 layers=4 heads=8 route_layers=2 || exit 1
variant largest width=192 layers=4 heads=8 route_layers=2 || exit 1

$PY -m ml summary capacity_current capacity_wider capacity_deeper capacity_route_head capacity_bigger \
    capacity_largest \
    > ml/runs/logs/capacity_probe_report.txt || exit 1
cat ml/runs/logs/capacity_probe_report.txt
echo "=== capacity probe finished"
