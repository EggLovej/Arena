#!/bin/sh
# Does learning from move scoring move the network off the plateau?
#
# Starting from overnight round 500: 10 iterations of distillation with move scoring as the teacher
# (value:PATH?candidates=8, which beat the plain network 64% and search 59%). Each iteration the
# current network, scoring moves, plays 150 games against itself; the network then learns its
# choices (3 epochs, small step size), and the next iteration's teacher is the new network.
#
# Then two measurements on fresh maps:
#   1. the plain network (no move scoring) at iterations 0, 5 and 10, rated against strategic_ai and
#      overnight rounds 300, 400 and 500, 100 games each: did the network itself improve?
#   2. move scoring on the final network against move scoring on round 500, the plain round 500,
#      and strategic_ai, 80 games each: did the teacher improve too?
#
# Expected time: about 1.5 hours. Safe to start again: finished steps are skipped.
# ml/runs/latest.pt (what the game's Neural AI plays) is put back as it was when the script ends.
#
# Run:    ml/scripts/done/value_distill_test.sh
# Watch:  .venv/bin/python -m ml monitor --run value_distill_seed0
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
START=ml/runs/overnight_scratch/round_0500.pt
RUN=value_distill_seed0
ITERATIONS=10

backup=$(mktemp)
if [ -f ml/runs/latest.pt ]; then cp ml/runs/latest.pt "$backup"; fi
trap 'if [ -s "$backup" ]; then cp "$backup" ml/runs/latest.pt; fi; rm -f "$backup"' EXIT

if [ ! -f ml/runs/$RUN/latest.pt ]; then
    echo "=== $RUN"
    $PY -u -m ml train distill --run $RUN --init $START --teacher value --value-candidates 8 \
        --iterations $ITERATIONS --games-per-iteration 150 --final-evaluation no --stop-after 5 --overwrite || exit 1
fi
until grep -q "\"round\": $ITERATIONS," ml/runs/$RUN/metrics.jsonl; do
    echo "=== $RUN (resuming)"
    $PY -u -m ml train distill --run $RUN --resume --stop-after 5 || exit 1
done

echo "=== the plain network, by iteration"
$PY -u -m ml compare --name value_distill \
    --run "value distill/seed 0=ml/runs/$RUN" --rounds 0 5 $ITERATIONS \
    --panel strategic_ai ml/runs/overnight_scratch/round_0300.pt ml/runs/overnight_scratch/round_0400.pt "$START" \
    --games 100 || exit 1

echo "=== move scoring on the final network"
FINAL=ml/runs/$RUN/round_$(printf %04d $ITERATIONS).pt
$PY -u -m ml evaluate --candidate "value:$FINAL?candidates=8" \
    --opponents "value:$START?candidates=8" "$START" strategic_ai \
    --generated-maps 40 --no-bundled-maps --skip-generated 30 || exit 1
echo "=== value distill test finished"
