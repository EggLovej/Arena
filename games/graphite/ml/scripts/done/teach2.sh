#!/bin/sh
# Teaching the policy again, now comparing moves instead of amount options (see plan.md,
# "Night of 2026-10-03"). Runs next to ml/scripts/done/critic_ab.sh.
#
# 1. Waits for the targets teach2 (started by hand: ml policy-targets --name teach2 --seed 101 ...).
# 2. Teaches at tau 0.01 (policy2_tau01) and at tau 0.003 (policy2_tau003), a stronger step.
# 3. Each on instinct against the network it was taught from: 100 fresh maps, both seats.
#
# Safe to start again: finished steps are skipped.
# Run:    nohup ml/scripts/done/teach2.sh > ml/runs/logs/teach2_chain_log.txt 2>&1 &
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
ORIGINAL=ml/runs/deep_from_imitation/round_0500.pt

while pgrep -f "policy-targets --name teach2" > /dev/null; do sleep 30; done
echo "=== targets teach2 ready ($(date +%H:%M))"

for tau in 01 003; do
    run=policy2_tau$tau
    if [ ! -f ml/runs/$run/summary.json ]; then
        echo "=== teaching $run"
        $PY -u -m ml policy-train --targets teach2 --run $run --tau 0.$tau --overwrite || exit 1
    fi
    if [ ! -f ml/runs/${run}_instinct.txt ]; then
        echo "=== $run on instinct vs the original ($(date +%H:%M))"
        $PY -u -m ml evaluate --candidate ml/runs/$run/best.pt --opponents $ORIGINAL \
            --generated-maps 100 --no-bundled-maps --skip-generated 30 --workers 4 > ml/runs/${run}_instinct.part || exit 1
        mv ml/runs/${run}_instinct.part ml/runs/${run}_instinct.txt
    fi
    tail -4 ml/runs/${run}_instinct.txt
done
echo "=== teach2 finished ($(date +%H:%M))"
