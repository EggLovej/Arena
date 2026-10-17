#!/bin/sh
# Step 1 of the user's plan (2026-10-11): free_scratch trains on until it stops improving.
#
# free_scratch (v2 inputs, free amounts, self-play from random weights) was still rising at round 300
# (1223, 1245, 1316 at rounds 100, 200, 300). The checks inside training cannot show a plateau: 16
# games against strategic_ai, which it already beats about 88% of the time. So it trains on in blocks
# of 100 rounds, the same run carried on exactly where it stopped (weights, optimizer, step size,
# entropy schedule: the bonus is 0.00048 at round 300, and its part above the floor 0.0002
# halves every 200 rounds). After each block its last round plays 200 games each on the
# comparison's fresh maps, both seats: the round 100 earlier, Neural AI's network
# (deep_from_imitation round 500, a fixed yardstick; both draw by chance here) and strategic_ai
# (`ml compare --name free_scratch_plateau_N`). The rule, set before any result: a block is flat
# when its last round scores under 55% against the round 100 earlier; two flat blocks in a row end
# the run. A cap at round 1500 (about 18 hours from round 300) ends it in any case.
# About 1.5 hours per block of training, then about 2 minutes for its matches.
# Resumable: run it again and it carries on where it stopped (finished matches are not replayed).
# ml/runs/latest.pt (the game's Neural AI) is set back to deep_from_imitation round 500 at the end.
#
# Run:    nohup ml/scripts/free_scratch_plateau.sh >> ml/runs/logs/free_scratch_log.txt 2>&1 < /dev/null &
# Watch:  .venv/bin/python -m ml tensorboard --run free_scratch
# Stop:   pkill -f free_scratch_plateau.sh; pkill -f "ml train selfplay --run free_scratch"
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
RUN=free_scratch
BLOCK=100
FIRST=400
LAST=1500
NEEDED=0.55
FLAT_IN_A_ROW=2
NEURAL=ml/runs/deep_from_imitation/round_0500.pt
trap 'cp ml/runs/deep_from_imitation/round_0500.pt ml/runs/latest.pt' EXIT

flat=0
end=$FIRST
while [ $end -le $LAST ]; do
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

    start=$((end - BLOCK))
    earlier=ml/runs/$RUN/round_$(printf %04d $start).pt
    name=free_scratch_plateau_$end
    if [ ! -f ml/runs/compare_$name/report.txt ]; then
        echo "=== round $end against round $start, Neural AI's network and strategic_ai ($(date +%H:%M))"
        $PY -u -m ml compare --name $name --run "free_scratch/seed 0=ml/runs/$RUN" --rounds $end \
            --panel strategic_ai $earlier $NEURAL --games 200 || exit 1
    fi
    score=$($PY -c "
import json
results = json.load(open('ml/runs/compare_$name/results.json'))['results']
print(next(points / games for _, opponent, points, games in results if opponent == '$earlier'))
") || exit 1
    if awk "BEGIN { exit !($score < $NEEDED) }"; then flat=$((flat + 1)); else flat=0; fi
    echo "=== round $end scored $score against round $start: $flat flat block(s) in a row ($(date +%H:%M))"
    if [ $flat -ge $FLAT_IN_A_ROW ]; then
        echo "=== free_scratch_plateau finished: flat from round $((end - FLAT_IN_A_ROW * BLOCK)) ($(date +%H:%M))"
        exit 0
    fi
    end=$((end + BLOCK))
done
echo "=== free_scratch_plateau finished: reached the cap, round $LAST, still rising ($(date +%H:%M))"
