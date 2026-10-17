#!/bin/sh
# After step 2 (the user's request, 2026-10-11): once free_gap_ramp.sh has finished, evaluate its
# result, then train the strongest network without a send limit at lookahead 0.995 for 150 rounds.
# Then a break: no judge training here.
#
# Evaluation. Every network of step 2 (free_gap10 every 50 rounds, then free_gap9 ... free_gap0 round
# 50) has played the control (free_scratch round 800) and Neural AI's network (deep_from_imitation
# round 500) on the same maps, limit lifted; their mean score is the yardstick. The best by it other
# than the ramp's end (free_gap0 round 50) plays the ramp's end head to head, 400 games, limit
# lifted. The strongest is the ramp's end if it scores at least 50% there, else the other. The
# strongest then plays Beast and strategic_ai, 200 games each, limit lifted (`compare_free_beast_check`).
# At 57.5% or more against Beast (+2.1 SE) it becomes Beast (`ml beast set`).
#
# The test arm: the strongest trains on without a send limit at lookahead 0.995 (0.99 so far) for 150
# rounds (run free_lookahead_trained, training maps from seed 12; the step size where the strongest
# stopped, the entropy bonus carrying on its schedule, no value-only rounds, no material hint).
# Rounds 50, 100 and 150 play the strongest, the control, Neural AI's network and strategic_ai, 200
# games each (`compare_free_lookahead_trained`). Its round with the best score against the strongest
# plays Beast and strategic_ai, 200 games each; the same 57.5% rule moves Beast. No 0.99 control arm
# (the user asked for this arm alone), so a gain over the start is not the lookahead's alone.
#
# Expected: about 2.5 hours after the ramp ends. Resumable; finished matches are not replayed.
# ml/runs/latest.pt (the game's Neural AI) is set back to deep_from_imitation round 500 at the end.
#
# Run:    nohup ml/scripts/free_after_ramp.sh >> ml/runs/logs/free_after_ramp_log.txt 2>&1 < /dev/null &
# Watch:  .venv/bin/python -m ml tensorboard --run free_lookahead_trained
# Stop:   pkill -f free_after_ramp.sh; pkill -f "ml train selfplay --run free_lookahead_trained"
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
CONTROL=ml/runs/free_scratch/round_0800.pt
CONTROL_ROUND=800
NEURAL=ml/runs/deep_from_imitation/round_0500.pt
RAMP_END=ml/runs/free_gap0/round_0050.pt
CROWN=0.575
RUN=free_lookahead_trained
ROUNDS=150
trap 'cp ml/runs/deep_from_imitation/round_0500.pt ml/runs/latest.pt' EXIT

# metrics round: the step size the run had reached at that round
step_size() {
    $PY -c "import json; print(next(r['learning rate'] for r in map(json.loads, open('$1')) if r['round'] == $2))"
}

# comparison opponent: the score of the comparison's checkpoint against that opponent
score() {
    $PY -c "
import json
results = json.load(open('ml/runs/compare_$1/results.json'))['results']
print(next(points / games for _, opponent, points, games in results if opponent == '$2'))"
}

# path: the run folder, and the round as a number
folder_of() { dirname $1; }
round_of() { expr $(basename $1 .pt | cut -d_ -f2) + 0; }

# comparison player: make the player Beast if it scored at least $CROWN against Beast there
crown_if_ahead() {
    against_beast=$(score $1 "$BEAST") || exit 1
    if awk "BEGIN { exit !($against_beast >= $CROWN) }"; then
        $PY -m ml beast set $2 --why "$(date +%F): beat Beast ($BEAST) $against_beast over 200 games, limit lifted (compare_$1)" || exit 1
        echo "=== $2 is the new Beast ($(date +%H:%M))"
    fi
}

until grep -q "=== free_gap_ramp finished" ml/runs/logs/free_gap_ramp_log.txt; do sleep 60; done
echo "=== free_gap_ramp has finished; evaluating ($(date +%H:%M))"

# --- The strongest of step 2 ---
CHALLENGER=$($PY -c "
import json, pathlib, sys
def scores(name):
    results = json.load(open(f'ml/runs/compare_{name}/results.json'))['results']
    return {opponent: points / games for _, opponent, points, games in results}
rows = []
for folder in sorted(pathlib.Path('ml/runs').glob('compare_free_gap10_*'), key=lambda f: int(f.name.rsplit('_', 1)[1])):
    end = int(folder.name.rsplit('_', 1)[1])
    rows.append((f'ml/runs/free_gap10/round_{end:04d}.pt', scores(folder.name.removeprefix('compare_'))))
for gap in (9, 8, 7, 6, 5, 4, 3, 2, 0):
    rows.append((f'ml/runs/free_gap{gap}/round_0050.pt', scores(f'free_ramp_gap{gap}')))
print('=== mean score against the control and Neural AI\'s network:', file=sys.stderr)
for path, score in rows:
    yardstick = (score['$CONTROL'] + score['$NEURAL']) / 2
    print(f'    {path}: {yardstick:.3f} ({score[\"$CONTROL\"]:.3f}, {score[\"$NEURAL\"]:.3f})', file=sys.stderr)
best = max((row for row in rows if row[0] != '$RAMP_END'), key=lambda row: row[1]['$CONTROL'] + row[1]['$NEURAL'])
print(best[0])") || exit 1
if [ ! -f ml/runs/compare_free_ramp_best/report.txt ]; then
    echo "=== the ramp's end against $CHALLENGER ($(date +%H:%M))"
    $PY -u -m ml compare --name free_ramp_best --unlimited --run "gap 0/seed 1=ml/runs/free_gap0" --rounds 50 \
        --panel strategic_ai $CHALLENGER --games 400 || exit 1
fi
against_challenger=$(score free_ramp_best $CHALLENGER) || exit 1
if awk "BEGIN { exit !($against_challenger >= 0.5) }"; then STRONGEST=$RAMP_END; else STRONGEST=$CHALLENGER; fi
echo "=== the ramp's end scored $against_challenger against $CHALLENGER: the strongest is $STRONGEST ($(date +%H:%M))"

BEAST=$($PY -c "from ml.play.beast import current; print(current()['player'])") || exit 1
if [ ! -f ml/runs/compare_free_beast_check/report.txt ]; then
    echo "=== $STRONGEST against Beast ($(date +%H:%M))"
    $PY -u -m ml compare --name free_beast_check --unlimited --run "strongest/$(basename $(folder_of $STRONGEST))=$(folder_of $STRONGEST)" \
        --rounds $(round_of $STRONGEST) --panel strategic_ai "$BEAST" --games 200 || exit 1
fi
echo "=== $STRONGEST, limit lifted: $(score free_beast_check "$BEAST") against Beast, $(score free_beast_check strategic_ai) against strategic_ai ($(date +%H:%M))"
crown_if_ahead free_beast_check $STRONGEST

# --- The test arm: lookahead 0.995 ---
if [ ! -f ml/runs/$RUN/latest.pt ]; then
    case $STRONGEST in
        ml/runs/free_gap10/*) BEHIND=$((CONTROL_ROUND + $(round_of $STRONGEST))) ;;
        *) LIMITED_END=$(sed -n 's/^=== gap 10 finished at round \([0-9]*\) .*/\1/p' ml/runs/logs/free_gap_ramp_log.txt | tail -1)
           steps=$(echo 9 8 7 6 5 4 3 2 0 | tr ' ' '\n' | grep -n "^$(basename $(folder_of $STRONGEST) | sed 's/free_gap//')$" | cut -d: -f1)
           BEHIND=$((CONTROL_ROUND + LIMITED_END + 50 * steps)) ;;
    esac
    STEP=$(step_size $(folder_of $STRONGEST)/metrics.jsonl $(round_of $STRONGEST)) || exit 1
    ENTROPY=$($PY -c "print(0.0002 + 0.0008 * 0.5 ** ($BEHIND / 200))") || exit 1
    echo "=== $RUN from $STRONGEST: lookahead 0.995, no send limit, seed 12, step size $STEP, entropy bonus $ENTROPY ($(date +%H:%M))"
    $PY -u -m ml train selfplay --run $RUN --init $STRONGEST --seed 12 --rounds $ROUNDS --min-send-gap 0 --lookahead 0.995 \
        --learning-rate $STEP --value-only-rounds 0 --material-hint 0 --evaluate-every 10 --final-evaluation no \
        --entropy-weight $ENTROPY --entropy-half-life 200 --entropy-floor 0.0002 --overwrite || exit 1
fi
until grep -q "\"round\": $ROUNDS," ml/runs/$RUN/metrics.jsonl; do
    echo "=== $RUN (resuming, $(date +%H:%M))"
    $PY -u -m ml train selfplay --run $RUN --resume || exit 1
done

if [ ! -f ml/runs/compare_$RUN/report.txt ]; then
    echo "=== $RUN with the strongest, the control and Neural AI's network ($(date +%H:%M))"
    $PY -u -m ml compare --name $RUN --unlimited --run "lookahead 0.995/seed 12=ml/runs/$RUN" --rounds 50 100 150 \
        --panel strategic_ai $STRONGEST $CONTROL $NEURAL --games 200 || exit 1
fi
BEST_ROUND=$($PY -c "
import json, sys
results = json.load(open('ml/runs/compare_$RUN/results.json'))['results']
rows = {}
for first, opponent, points, games in results:
    rows.setdefault(int(first.rsplit(' ', 1)[1]), {})[opponent] = points / games
for round_number, score in sorted(rows.items()):
    print(f'=== $RUN round {round_number}, limit lifted: {score[\"$STRONGEST\"]} against the strongest, {score[\"$CONTROL\"]} against the control, {score[\"$NEURAL\"]} against Neural AI\'s network, {score[\"strategic_ai\"]} against strategic_ai', file=sys.stderr)
print(max(sorted(rows), key=lambda round_number: rows[round_number]['$STRONGEST']))") || exit 1

BEAST=$($PY -c "from ml.play.beast import current; print(current()['player'])") || exit 1
if [ ! -f ml/runs/compare_${RUN}_beast/report.txt ]; then
    echo "=== $RUN round $BEST_ROUND against Beast ($(date +%H:%M))"
    $PY -u -m ml compare --name ${RUN}_beast --unlimited --run "lookahead 0.995/seed 12=ml/runs/$RUN" --rounds $BEST_ROUND \
        --panel strategic_ai "$BEAST" --games 200 || exit 1
fi
echo "=== $RUN round $BEST_ROUND, limit lifted: $(score ${RUN}_beast "$BEAST") against Beast, $(score ${RUN}_beast strategic_ai) against strategic_ai ($(date +%H:%M))"
crown_if_ahead ${RUN}_beast ml/runs/$RUN/round_$(printf %04d $BEST_ROUND).pt
echo "=== free_after_ramp finished ($(date +%H:%M))"
