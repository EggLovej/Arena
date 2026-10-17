#!/bin/sh
# A second round of the send-limit ramp (the user's request, 2026-10-12), then its evaluation.
#
# The start and the control: free_gap0 round 50, the end of the first ramp (free_gap_ramp.sh) and
# Beast (at temperature 0.2). Everything as in the first round: the step size where it stopped
# (2.0e-05), the entropy bonus carrying on its schedule (about at its floor, 0.0002, by now), no
# value-only rounds, no material hint; the optimizer's memory starts afresh.
#
# Phase 1, at most one send a second (min_send_gap 10, run free2_gap10, training maps from seed 210),
# in blocks of 50 rounds. After each block its last round, with the limit lifted, plays 200 games
# each on the comparison's fresh maps, both seats: the round 50 earlier (the control for the first
# block), the control, Neural AI's network (deep_from_imitation round 500) and strategic_ai
# (`ml compare --unlimited --name free2_gap10_N`, drawing by chance as every comparison). A block is
# flat when it scores under 55% against the round 50 earlier; two flat blocks in a row end the phase,
# round 300 at the latest.
#
# Phase 2, from phase 1's last round: 50 rounds each at gaps 9, 8, 7, 6, 5, 4, 3, 2, then none
# (runs free2_gap9 ... free2_gap2, free2_gap0; seeds 209 ... 202, 201). After each step its last
# round, limit lifted, plays phase 1's last round, the control, Neural AI's network and strategic_ai,
# 200 games each (`ml compare --unlimited --name free2_ramp_gapN`).
#
# Evaluation (the user's design): the best network of this round by the yardstick of the first round
# (mean score against the control and Neural AI's network), other than the final one (free2_gap0
# round 50), the final one and free_gap0 round 50 play each other and strategic_ai, 200 games per
# pairing, 1200 in all, at temperature 0.2 as Beast plays, no send limit
# (ml/scripts/round_robin.py --name free2; ml/runs/logs/round_robin_free2.json).
#
# Expected: about 8 hours. Resumable: run it again and it carries on where it stopped.
# ml/runs/latest.pt (the game's Neural AI) is set back to deep_from_imitation round 500 at the end.
#
# Run:    nohup ml/scripts/free_gap_ramp2.sh >> ml/runs/logs/free_gap_ramp2_log.txt 2>&1 < /dev/null &
# Watch:  .venv/bin/python -m ml tensorboard --run free2_gap10   (then free2_gap9, ...)
# Stop:   pkill -f free_gap_ramp2.sh; pkill -f "ml train selfplay --run free2_gap"
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
CONTROL=ml/runs/free_gap0/round_0050.pt
CONTROL_METRICS=ml/runs/free_gap0/metrics.jsonl
CONTROL_ROUND=50
CONTROL_BEHIND=1500  # rounds of the entropy schedule behind the control: 800 + 250 + 9 x 50
NEURAL=ml/runs/deep_from_imitation/round_0500.pt
BLOCK=50
LIMITED_MOST=300
NEEDED=0.55
FLAT_IN_A_ROW=2
STEP_ROUNDS=50
trap 'cp ml/runs/deep_from_imitation/round_0500.pt ml/runs/latest.pt' EXIT

# run gap seed start step-size entropy-bonus rounds
start_run() {
    echo "=== $1 from $4: gap $2, seed $3, step size $5, entropy bonus $6 ($(date +%H:%M))"
    $PY -u -m ml train selfplay --run $1 --init $4 --seed $3 --rounds $7 --min-send-gap $2 \
        --learning-rate $5 --value-only-rounds 0 --material-hint 0 --evaluate-every 10 --final-evaluation no \
        --entropy-weight $6 --entropy-half-life 200 --entropy-floor 0.0002 --overwrite
}

# metrics round: the step size the run had reached at that round
step_size() {
    $PY -c "import json; print(next(r['learning rate'] for r in map(json.loads, open('$1')) if r['round'] == $2))"
}

# rounds of the entropy schedule behind us
entropy_after() {
    $PY -c "print(0.0002 + 0.0008 * 0.5 ** ($1 / 200))"
}

# comparison opponent: the score of the comparison's checkpoint against that opponent
score() {
    $PY -c "
import json
results = json.load(open('ml/runs/compare_$1/results.json'))['results']
print(next(points / games for _, opponent, points, games in results if opponent == '$2'))"
}

# --- Phase 1: gap 10 ---
RUN=free2_gap10
flat=0
end=$BLOCK
while [ $end -le $LIMITED_MOST ]; do
    if [ ! -f ml/runs/$RUN/latest.pt ]; then
        STEP=$(step_size $CONTROL_METRICS $CONTROL_ROUND) || exit 1
        ENTROPY=$(entropy_after $CONTROL_BEHIND) || exit 1
        start_run $RUN 10 210 $CONTROL $STEP $ENTROPY $end || exit 1
    fi
    $PY -c "
import json, pathlib
path = pathlib.Path('ml/runs/$RUN/settings.json')
settings = json.loads(path.read_text())
settings['rounds'] = max(settings['rounds'], $end)
path.write_text(json.dumps(settings, indent=2) + '\n')
" || exit 1
    until grep -q "\"round\": $end," ml/runs/$RUN/metrics.jsonl; do
        echo "=== $RUN to round $end (resuming, $(date +%H:%M))"
        $PY -u -m ml train selfplay --run $RUN --resume || exit 1
    done

    if [ $end -eq $BLOCK ]; then
        earlier=$CONTROL
        panel="strategic_ai $CONTROL $NEURAL"
    else
        earlier=ml/runs/$RUN/round_$(printf %04d $((end - BLOCK))).pt
        panel="strategic_ai $earlier $CONTROL $NEURAL"
    fi
    name=free2_gap10_$end
    if [ ! -f ml/runs/compare_$name/report.txt ]; then
        echo "=== $RUN round $end with the limit lifted ($(date +%H:%M))"
        $PY -u -m ml compare --name $name --unlimited --run "round 2 gap 10/seed 210=ml/runs/$RUN" --rounds $end \
            --panel $panel --games 200 || exit 1
    fi
    against_earlier=$(score $name $earlier) || exit 1
    against_control=$(score $name $CONTROL) || exit 1
    against_neural=$(score $name $NEURAL) || exit 1
    if awk "BEGIN { exit !($against_earlier < $NEEDED) }"; then flat=$((flat + 1)); else flat=0; fi
    echo "=== $RUN round $end, limit lifted: $against_earlier against the round $BLOCK earlier, $against_control against the control, $against_neural against Neural AI's network; $flat flat block(s) in a row ($(date +%H:%M))"
    LIMITED_END=$end
    [ $flat -ge $FLAT_IN_A_ROW ] && break
    end=$((end + BLOCK))
done
echo "=== gap 10 finished at round $LIMITED_END ($(date +%H:%M))"

# --- Phase 2: the limit lifted step by step ---
RAMP_START=ml/runs/free2_gap10/round_$(printf %04d $LIMITED_END).pt
START=$RAMP_START
PREVIOUS_METRICS=ml/runs/free2_gap10/metrics.jsonl
PREVIOUS_ROUND=$LIMITED_END
BEHIND=$((CONTROL_BEHIND + LIMITED_END))
for GAP in 9 8 7 6 5 4 3 2 0; do
    RUN=free2_gap$GAP
    SEED=$((200 + GAP))
    [ $GAP -eq 0 ] && SEED=201
    if ! grep -q "\"round\": $STEP_ROUNDS," ml/runs/$RUN/metrics.jsonl 2>/dev/null; then
        if [ -f ml/runs/$RUN/latest.pt ]; then
            echo "=== $RUN (resuming, $(date +%H:%M))"
            $PY -u -m ml train selfplay --run $RUN --resume || exit 1
        else
            STEP=$(step_size $PREVIOUS_METRICS $PREVIOUS_ROUND) || exit 1
            ENTROPY=$(entropy_after $BEHIND) || exit 1
            start_run $RUN $GAP $SEED $START $STEP $ENTROPY $STEP_ROUNDS || exit 1
        fi
    fi
    name=free2_ramp_gap$GAP
    if [ ! -f ml/runs/compare_$name/report.txt ]; then
        echo "=== $RUN with the limit lifted ($(date +%H:%M))"
        $PY -u -m ml compare --name $name --unlimited --run "round 2 gap $GAP/seed $SEED=ml/runs/$RUN" --rounds $STEP_ROUNDS \
            --panel strategic_ai $RAMP_START $CONTROL $NEURAL --games 200 || exit 1
    fi
    echo "=== $RUN round $STEP_ROUNDS, limit lifted: $(score $name $RAMP_START) against the ramp's start, $(score $name $CONTROL) against the control, $(score $name $NEURAL) against Neural AI's network ($(date +%H:%M))"
    START=ml/runs/$RUN/round_$(printf %04d $STEP_ROUNDS).pt
    PREVIOUS_METRICS=ml/runs/$RUN/metrics.jsonl
    PREVIOUS_ROUND=$STEP_ROUNDS
    BEHIND=$((BEHIND + STEP_ROUNDS))
done
echo "=== the second ramp finished ($(date +%H:%M))"

# --- Evaluation: the best, the final one and free_gap0 round 50 ---
FINAL=ml/runs/free2_gap0/round_0050.pt
BEST=$($PY -c "
import json, pathlib, sys
def scores(name):
    results = json.load(open(f'ml/runs/compare_{name}/results.json'))['results']
    return {opponent: points / games for _, opponent, points, games in results}
rows = []
for folder in sorted(pathlib.Path('ml/runs').glob('compare_free2_gap10_*'), key=lambda f: int(f.name.rsplit('_', 1)[1])):
    end = int(folder.name.rsplit('_', 1)[1])
    rows.append((f'ml/runs/free2_gap10/round_{end:04d}.pt', scores(folder.name.removeprefix('compare_'))))
for gap in (9, 8, 7, 6, 5, 4, 3, 2, 0):
    rows.append((f'ml/runs/free2_gap{gap}/round_0050.pt', scores(f'free2_ramp_gap{gap}')))
print('=== mean score against the control and Neural AI\'s network (limit lifted, by chance):', file=sys.stderr)
for path, score in rows:
    yardstick = (score['$CONTROL'] + score['$NEURAL']) / 2
    print(f'    {path}: {yardstick:.3f} ({score[\"$CONTROL\"]:.3f}, {score[\"$NEURAL\"]:.3f})', file=sys.stderr)
best = max((row for row in rows if row[0] != '$FINAL'), key=lambda row: row[1]['$CONTROL'] + row[1]['$NEURAL'])
print(best[0])") || exit 1
best_label="$(basename $(dirname $BEST)) r$(expr $(basename $BEST .pt | cut -d_ -f2) + 0) (best)"
echo "=== the best other than the final one: $BEST ($(date +%H:%M))"
if [ ! -f ml/runs/logs/round_robin_free2.json ]; then
    $PY -u ml/scripts/round_robin.py --name free2 --temperature 0.2 --games 200 --opponents strategic_ai \
        --network "free_gap0 r50 (now)=$CONTROL" --network "$best_label=$BEST" \
        --network "free2_gap0 r50 (final)=$FINAL" || exit 1
fi
echo "=== free_gap_ramp2 finished ($(date +%H:%M))"
