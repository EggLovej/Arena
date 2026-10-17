#!/bin/sh
# Judge as critic: does self-play learn faster when moves are judged against the judge's expectations?
#
# In self-play, a move's advantage is "how things went afterwards, compared with what was expected
# before". Normally the network's own value output says what was expected; it learns from one
# round's games at a time and moves with every round. judge2 learned from 37,500 games and is
# frozen, so its expectations should be steadier and sharper, and the advantages less noisy.
#
# Both arms continue deep_from_imitation from round 500 with identical settings (the step size it
# ended with, no value-only rounds, no material hint), the same seed and so the same first maps:
#
#   plain    as before: expectations from the network's value output
#   critic   expectations from judge2 (--critic)
#
# Then one comparison on fresh maps: rounds 0, 50, 100 and 150 of both arms against strategic_ai,
# aggressive_ai and deep_from_imitation round 400, 100 games each, and the final rounds head to head.
#
# Expected time: about 4.5 hours. Safe to start again: finished steps are skipped.
# ml/runs/latest.pt (what the game's Neural AI plays) is put back as it was when the script ends.
#
# Run:    nohup ml/scripts/done/critic_ab.sh > ml/runs/logs/critic_ab_log.txt 2>&1 &
# Watch:  .venv/bin/python -m ml monitor --run ab_critic   (or ab_plain)
# Stop:   pkill -f critic_ab.sh; pkill -f "ml train"
# Result: ml/runs/compare_critic_ab/
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
START=ml/runs/deep_from_imitation/round_0500.pt
ROUNDS=${ROUNDS:-150}
SHARED="--learning-rate 4.4e-5 --value-only-rounds 0 --material-hint 0 --evaluate-every 10 --final-evaluation no"

backup=$(mktemp)
if [ -f ml/runs/latest.pt ]; then cp ml/runs/latest.pt "$backup"; fi
trap 'if [ -s "$backup" ]; then cp "$backup" ml/runs/latest.pt; fi; rm -f "$backup"' EXIT

arm() {  # run name, extra settings
    run=$1
    shift
    if [ ! -f ml/runs/$run/latest.pt ]; then
        echo "=== $run"
        $PY -u -m ml train selfplay --run $run --init $START --seed 0 --rounds $ROUNDS $SHARED "$@" --stop-after 50 --overwrite || return 1
    fi
    until grep -q "\"round\": $ROUNDS," ml/runs/$run/metrics.jsonl; do
        echo "=== $run (resuming)"
        $PY -u -m ml train selfplay --run $run --resume --stop-after 50 || return 1
    done
    echo "=== $run finished"
}

arm ab_plain || exit 1
arm ab_critic --critic ml/runs/judge2/best.pt || exit 1

echo "=== comparison"
$PY -u -m ml compare --name critic_ab \
    --run "plain/seed 0=ml/runs/ab_plain" \
    --run "critic/seed 0=ml/runs/ab_critic" \
    --rounds 0 50 100 150 \
    --panel strategic_ai aggressive_ai ml/runs/deep_from_imitation/round_0400.pt \
    --games 100 || exit 1
echo "=== critic A/B finished"
