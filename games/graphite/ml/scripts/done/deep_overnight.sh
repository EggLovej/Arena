#!/bin/sh
# The long run with the deeper network (64 wide, 6 layers; the capacity probe's winner), plain
# self-play with the default settings, exactly like the first overnight run otherwise.
#
# 1. From random weights: 500 rounds (deep_scratch), evaluation every 10 rounds.
# 2. Its rating curve next to the first overnight run (64 wide, 3 layers): rounds 0-500 of both on
#    one scale, against strategic_ai, 100 games each, and the two final rounds head to head.
# 3. If the night is long enough: the same from an imitation start (deep_imitation, 20 epochs of
#    copying strategic_ai, then deep_from_imitation, 500 rounds). This is Stage 10's question on
#    the network we keep. Stop it in the morning if it is still running; it resumes where it was.
#
# Each training process stops after 50 rounds and the next one resumes, as in the first overnight run.
# Expected time: about 5 hours for step 1, 20 minutes for step 2, 5 more hours for step 3.
#
# Run:    ml/scripts/done/deep_overnight.sh
# Watch:  .venv/bin/python -m ml monitor --run deep_scratch
# Stop:   pkill -f deep_overnight.sh; pkill -f "ml train"
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
ROUNDS=500

selfplay() {  # run name, starting checkpoint
    if [ ! -f ml/runs/$1/latest.pt ]; then
        echo "=== $1"
        $PY -u -m ml train selfplay --run $1 --init $2 --rounds $ROUNDS --evaluate-every 10 --stop-after 50 --overwrite || return 1
    fi
    until grep -q "\"round\": $ROUNDS," ml/runs/$1/metrics.jsonl; do
        echo "=== $1 (resuming)"
        $PY -u -m ml train selfplay --run $1 --resume --stop-after 50 || return 1
    done
    echo "=== $1 finished"
}

mkdir -p ml/runs/deep_starts
DEEP=ml/runs/deep_starts/random_seed0.pt
[ -f $DEEP ] || $PY -m ml checkpoint new $DEEP --seed 0 --set layers=6 || exit 1

# 1. From random weights.
selfplay deep_scratch $DEEP || exit 1

# 2. Next to the first overnight run.
if [ ! -f ml/runs/compare_deep_vs_overnight/report.txt ]; then
    echo "=== comparison with the first overnight run"
    $PY -u -m ml compare --name deep_vs_overnight \
        --run "64x3/seed 0=ml/runs/overnight_scratch" \
        --run "64x6/seed 0=ml/runs/deep_scratch" \
        --rounds 0 50 100 200 300 400 500 \
        --panel strategic_ai ml/runs/overnight_scratch/round_0500.pt \
        --games 100 || exit 1
fi

# 3. From an imitation start.
if ! grep -q '"round": 20,' ml/runs/deep_imitation/metrics.jsonl 2>/dev/null; then
    echo "=== deep_imitation"
    $PY -u -m ml train imitation --run deep_imitation --init $DEEP --seed 0 --evaluation-maps 0 --final-evaluation no --overwrite || exit 1
fi
selfplay deep_from_imitation ml/runs/deep_imitation/round_0020.pt || exit 1
echo "=== deep overnight finished"
