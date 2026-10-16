#!/bin/sh
# Step 2 of the user's plan (2026-10-11): the send limit on free_scratch, which stopped improving, then
# lifted step by step (as "Acting once a second" did for the old design, where it was the one thing
# that got past a plateau: about +80).
#
# The start and the control: free_scratch round 800 (flat from round 600; about 42% against Neural
# AI's network from round 400 on). Every run here keeps free_scratch's settings except what a run
# started from a trained network needs: no value-only rounds, no material hint, the step size where
# free_scratch stopped (3.0e-05), and the entropy bonus carrying on its schedule (0.00025 at round 800,
# its part above the floor 0.0002 halving every 200 rounds). The optimizer's memory starts afresh.
#
# Phase 1, at most one send a second (min_send_gap 10, run free_gap10, training maps from seed 10), in
# blocks of 50 rounds. After each block its last round, with the limit lifted, plays 200 games each on
# the comparison's fresh maps, both seats: the round 50 earlier (also lifted; for the first block that
# is the control), the control, Neural AI's network (deep_from_imitation round 500) and strategic_ai
# (`ml compare --unlimited --name free_gap10_N`). The rule, set before any result: a block is flat
# when it scores under 55% against the round 50 earlier; two flat blocks in a row end the phase,
# round 300 at the latest.
#
# Phase 2, the limit lifted step by step from phase 1's last round: 50 rounds each at gaps 9, 8, 7, 6,
# 5, 4, 3 and 2 ticks, then 50 with no limit at all (gap 1 is no limit: one send per tick is all the
# game takes). Each step is its own run (free_gap9 ... free_gap2, free_gap0), so every tool knows the
# limit a network trained under; training maps from seed = gap (seed 1 for no limit). After each step
# its last round, limit lifted, plays phase 1's last round, the control, Neural AI's network and
# strategic_ai, 200 games each (`ml compare --unlimited --name free_ramp_gapN`).
#
# Expected time: about 6 to 10 hours, then about 3 minutes for each block's or step's matches.
# Resumable: run it again and it carries on where it stopped (finished matches are not replayed).
# ml/runs/latest.pt (the game's Neural AI) is set back to deep_from_imitation round 500 at the end.
#
# Run:    nohup ml/scripts/free_gap_ramp.sh >> ml/runs/logs/free_gap_ramp_log.txt 2>&1 < /dev/null &
# Watch:  .venv/bin/python -m ml tensorboard --run free_gap10   (then free_gap9, ...)
# Stop:   pkill -f free_gap_ramp.sh; pkill -f "ml train selfplay --run free_gap"
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
CONTROL=ml/runs/free_scratch/round_0800.pt
CONTROL_ROUND=800
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
RUN=free_gap10
flat=0
end=$BLOCK
while [ $end -le $LIMITED_MOST ]; do
    if [ ! -f ml/runs/$RUN/latest.pt ]; then
        STEP=$(step_size ml/runs/free_scratch/metrics.jsonl $CONTROL_ROUND) || exit 1
        ENTROPY=$(entropy_after $CONTROL_ROUND) || exit 1
        start_run $RUN 10 10 $CONTROL $STEP $ENTROPY $end || exit 1
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
    name=free_gap10_$end
    if [ ! -f ml/runs/compare_$name/report.txt ]; then
        echo "=== $RUN round $end with the limit lifted ($(date +%H:%M))"
        $PY -u -m ml compare --name $name --unlimited --run "gap 10/seed 10=ml/runs/$RUN" --rounds $end \
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
RAMP_START=ml/runs/free_gap10/round_$(printf %04d $LIMITED_END).pt
START=$RAMP_START
PREVIOUS_METRICS=ml/runs/free_gap10/metrics.jsonl
PREVIOUS_ROUND=$LIMITED_END
BEHIND=$((CONTROL_ROUND + LIMITED_END))
for GAP in 9 8 7 6 5 4 3 2 0; do
    RUN=free_gap$GAP
    SEED=$GAP
    [ $GAP -eq 0 ] && SEED=1
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
    name=free_ramp_gap$GAP
    if [ ! -f ml/runs/compare_$name/report.txt ]; then
        echo "=== $RUN with the limit lifted ($(date +%H:%M))"
        $PY -u -m ml compare --name $name --unlimited --run "gap $GAP/seed $SEED=ml/runs/$RUN" --rounds $STEP_ROUNDS \
            --panel strategic_ai $RAMP_START $CONTROL $NEURAL --games 200 || exit 1
    fi
    echo "=== $RUN round $STEP_ROUNDS, limit lifted: $(score $name $RAMP_START) against the ramp's start, $(score $name $CONTROL) against the control, $(score $name $NEURAL) against Neural AI's network ($(date +%H:%M))"
    START=ml/runs/$RUN/round_$(printf %04d $STEP_ROUNDS).pt
    PREVIOUS_METRICS=ml/runs/$RUN/metrics.jsonl
    PREVIOUS_ROUND=$STEP_ROUNDS
    BEHIND=$((BEHIND + STEP_ROUNDS))
done
echo "=== free_gap_ramp finished ($(date +%H:%M))"
