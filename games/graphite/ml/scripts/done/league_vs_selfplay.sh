#!/bin/sh
# League against plain self-play, from the overnight model's plateau.
#
# The overnight run (plain self-play, only ever against its current self) stopped improving after
# about round 300. Does a league get it moving again? Both arms start from round 500 and train
# 50 more rounds with identical settings; the only difference is who the network plays:
#
#   selfplay   every game against itself, as overnight
#   league     half against itself, 30% against past champions, 20% against the scripted AIs,
#              with a promotion match against the reigning champion every 5 rounds
#
# Both arms continue the overnight run as it ended: the step size it had reached (5.9e-5), no
# value-only rounds and no material hint (both only make sense for a fresh network). Two seeds
# per arm, because two runs with identical settings differ by up to about 50 rating points.
# The league also plays 128 promotion games every 5 rounds; those are not trained on.
#
# Then one comparison on fresh maps: rounds 0, 25 and 50 of every run (round 0 is round 500 of
# the overnight run in all four, so its four ratings also show the measuring noise) against
# strategic_ai and overnight rounds 300, 400 and 500, 100 games each, and the final rounds of
# the two arms head to head.
#
# Expected time: about 2 hours. Safe to start again: finished steps are skipped.
# ml/runs/latest.pt (what the game's Neural AI plays) is put back as it was when the script ends.
#
# Run:    ml/scripts/done/league_vs_selfplay.sh
# Watch:  .venv/bin/python -m ml monitor --run lvs_league_seed0   (or any run below)
# Stop:   pkill -f league_vs_selfplay.sh; pkill -f "ml train"
# Result: ml/runs/compare_league_vs_selfplay/
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
START=ml/runs/overnight_scratch/round_0500.pt
ROUNDS=50
SEEDS="0 1"
SHARED="--rounds $ROUNDS --learning-rate 5.9e-5 --value-only-rounds 0 --material-hint 0 --evaluate-every 10 --final-evaluation no"

backup=$(mktemp)
if [ -f ml/runs/latest.pt ]; then cp ml/runs/latest.pt "$backup"; fi
trap 'if [ -s "$backup" ]; then cp "$backup" ml/runs/latest.pt; fi; rm -f "$backup"' EXIT

arm() {  # trainer, run name, seed
    if [ ! -f ml/runs/$2/latest.pt ]; then
        echo "=== $2"
        $PY -u -m ml train $1 --run $2 --init $START --seed $3 $SHARED --stop-after 50 --overwrite || return 1
    fi
    until grep -q "\"round\": $ROUNDS," ml/runs/$2/metrics.jsonl; do
        echo "=== $2 (resuming)"
        $PY -u -m ml train $1 --run $2 --resume --stop-after 50 || return 1
    done
    echo "=== $2 finished"
}

for seed in $SEEDS; do
    arm selfplay lvs_selfplay_seed$seed $seed || exit 1
    arm league lvs_league_seed$seed $seed || exit 1
done

echo "=== comparison"
$PY -u -m ml compare --name league_vs_selfplay \
    --run "selfplay/seed 0=ml/runs/lvs_selfplay_seed0" \
    --run "selfplay/seed 1=ml/runs/lvs_selfplay_seed1" \
    --run "league/seed 0=ml/runs/lvs_league_seed0" \
    --run "league/seed 1=ml/runs/lvs_league_seed1" \
    --rounds 0 25 50 \
    --panel strategic_ai ml/runs/overnight_scratch/round_0300.pt ml/runs/overnight_scratch/round_0400.pt \
        ml/runs/overnight_scratch/round_0500.pt \
    --games 100 || exit 1
echo "=== league vs self-play finished"
