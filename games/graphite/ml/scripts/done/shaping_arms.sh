#!/bin/sh
# Four ways to give self-play a clearer learning signal, against plain self-play (2026-10-04).
#
# Self-play from round 900 no longer improves, and its networks pick amounts almost at random
# and send about twice as often as the imitation start. Four arms, each continuing round 900
# (ab_plain round 400) for 150 rounds with the settings of the control, small_selfplay (already
# run on 2026-10-03: step size 4.4e-5, no value-only rounds, no material hint, seed 0):
#
#   shaping_entropy      entropy bonus 0.001 instead of 0.01 (is the bonus what keeps the amounts random?)
#   shaping_production   production potential 0.5: credit for every change in my share of the
#                        map's production (captures, defences, losses)
#   shaping_front        front potential 0.5: credit for moving units towards the enemy, cost for
#                        pulling them back (ml.training.selfplay.front_potential)
#   shaping_all          all three together
#
# Then ml compare of rounds 0, 50, 100, 150 of all five runs (round 0 is round 900 in every run,
# so its five ratings show the measuring noise) on the usual panel, the last rounds head to head,
# and ml behaviour of the last rounds (where the sends go, how decided the amounts are).
#
# Expected time: about 1 h 50 min per arm, 40 minutes for the comparison: about 8 hours.
# Safe to start again: finished steps are skipped. ml/runs/latest.pt (the game's Neural AI) is set
# back to deep_from_imitation round 500 when the script ends.
#
# Run:    nohup ml/scripts/done/shaping_arms.sh > ml/runs/logs/shaping_arms_log.txt 2>&1 &
# Watch:  .venv/bin/python -m ml monitor --run shaping_all   (or any arm)
# Stop:   pkill -f shaping_arms.sh; pkill -f "ml train"
# Result: ml/runs/compare_shaping/report.txt and ml/runs/logs/shaping_behaviour.txt
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
START=ml/runs/ab_plain/round_0400.pt
ROUNDS=150
SHARED="--learning-rate 4.4e-5 --value-only-rounds 0 --material-hint 0 --evaluate-every 10 --final-evaluation no"
trap 'cp ml/runs/deep_from_imitation/round_0500.pt ml/runs/latest.pt' EXIT

arm() {  # run name, extra settings
    run=$1
    shift
    if [ ! -f ml/runs/$run/latest.pt ]; then
        echo "=== $run ($(date +%H:%M))"
        $PY -u -m ml train selfplay --run $run --init $START --seed 0 --rounds $ROUNDS $SHARED "$@" --stop-after 50 --overwrite || return 1
    fi
    until grep -q "\"round\": $ROUNDS," ml/runs/$run/metrics.jsonl; do
        echo "=== $run (resuming, $(date +%H:%M))"
        $PY -u -m ml train selfplay --run $run --resume --stop-after 50 || return 1
    done
    echo "=== $run finished ($(date +%H:%M))"
}

arm shaping_entropy --entropy-weight 0.001 || exit 1
arm shaping_production --production-potential 0.5 || exit 1
arm shaping_front --front-potential 0.5 || exit 1
arm shaping_all --entropy-weight 0.001 --production-potential 0.5 --front-potential 0.5 || exit 1

echo "=== comparison ($(date +%H:%M))"
$PY -u -m ml compare --name shaping \
    --run "plain/seed 0=ml/runs/small_selfplay" \
    --run "entropy 0.001/seed 0=ml/runs/shaping_entropy" \
    --run "production/seed 0=ml/runs/shaping_production" \
    --run "front/seed 0=ml/runs/shaping_front" \
    --run "all three/seed 0=ml/runs/shaping_all" \
    --rounds 0 50 100 150 \
    --panel strategic_ai aggressive_ai ml/runs/deep_from_imitation/round_0400.pt ml/runs/deep_scratch/round_0300.pt \
    --games 100 || exit 1

echo "=== behaviour ($(date +%H:%M))"
$PY -u -m ml behaviour $START ml/runs/small_selfplay/round_0150.pt ml/runs/shaping_entropy/round_0150.pt \
    ml/runs/shaping_production/round_0150.pt ml/runs/shaping_front/round_0150.pt ml/runs/shaping_all/round_0150.pt \
    > ml/runs/logs/shaping_behaviour.txt || exit 1
cat ml/runs/logs/shaping_behaviour.txt
echo "=== shaping arms finished ($(date +%H:%M))"
