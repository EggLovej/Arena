#!/bin/sh
# Stage 10: random start against imitation start.
# Three seeds per route, 150 rounds of self-play each with the default settings, then one comparison on fresh maps.
# Seed 0 of the random start is the first 150 rounds of the overnight run; seed 0 of the imitation start begins
# from imitation_v1. Safe to start again: every step skips what is already finished.
#
# Watch a run:  .venv/bin/python -m ml monitor --run stage10_imitation_start_seed0   (or any other run name below)
# Stop:         pkill -f stage10.sh; pkill -f "ml train"
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
ROUNDS=150

imitation() {  # seed
    run=stage10_imitation_seed$1
    if ! grep -q '"round": 20,' ml/runs/$run/metrics.jsonl 2>/dev/null; then
        $PY -u -m ml train imitation --run $run --seed $1 --evaluation-maps 0 --final-evaluation no --overwrite || return 1
    fi
}

selfplay() {  # run name, starting point, seed
    if [ ! -f ml/runs/$1/latest.pt ]; then
        $PY -u -m ml train selfplay --run $1 --init $2 --seed $3 --rounds $ROUNDS --evaluate-every 10 \
            --final-evaluation no --stop-after 50 --overwrite || return 1
    fi
    until grep -q "\"round\": $ROUNDS," ml/runs/$1/metrics.jsonl; do
        $PY -u -m ml train selfplay --run $1 --resume --stop-after 50 || return 1
    done
}

imitation 1 || exit 1
imitation 2 || exit 1
selfplay stage10_imitation_start_seed0 ml/runs/imitation_v1/round_0020.pt 0 || exit 1
selfplay stage10_random_start_seed1 random 1 || exit 1
selfplay stage10_imitation_start_seed1 ml/runs/stage10_imitation_seed1/round_0020.pt 1 || exit 1
selfplay stage10_random_start_seed2 random 2 || exit 1
selfplay stage10_imitation_start_seed2 ml/runs/stage10_imitation_seed2/round_0020.pt 2 || exit 1

$PY -u -m ml compare --name stage10 \
    --run "random start/seed 0=ml/runs/overnight_scratch" \
    --run "random start/seed 1=ml/runs/stage10_random_start_seed1" \
    --run "random start/seed 2=ml/runs/stage10_random_start_seed2" \
    --run "imitation start/seed 0=ml/runs/stage10_imitation_start_seed0" \
    --run "imitation start/seed 1=ml/runs/stage10_imitation_start_seed1" \
    --run "imitation start/seed 2=ml/runs/stage10_imitation_start_seed2" \
    --rounds 0 25 50 100 150 \
    --panel aggressive_ai careful_ai strategic_ai ml/runs/overnight_scratch/round_0100.pt \
        ml/runs/overnight_scratch/round_0200.pt ml/runs/overnight_scratch/round_0300.pt \
        ml/runs/overnight_scratch/round_0500.pt \
    --games 100 || exit 1
echo "=== stage 10 finished"
