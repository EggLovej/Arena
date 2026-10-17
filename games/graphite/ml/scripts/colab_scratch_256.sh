#!/bin/sh
# On a Colab runtime: a 256x8 network (4.5 million weights) from random weights, by self-play only
# (no imitation, no kickstart). Started through ml/colab/scratch_256.ipynb (ml colab run).
#
# - 500 rounds, settings as deep_scratch (the 64x6 network from random weights, 500 rounds), except:
#   the entropy bonus starts at 0.01 and halves every 100 rounds towards a floor of 0.0005, and
#   the networks are served from the GPU (on the A100 that keeps rounds at about 47 s at this size).
# - Trains in chunks of 50 rounds. After each, the files worth keeping (metrics, settings, latest,
#   best, every 50th round) are copied to Google Drive ($DRIVE/runs/scratch_256). A new runtime
#   starts from there: running this script again carries on where the last one stopped.
# - At the end: ml compare of rounds 0, 100, ..., 500 of this run and of deep_scratch (64x6,
#   entropy 0.01) on one scale, on the GPU; the report goes to Drive too.
#
# Expected: about 8-11 hours on an A100 (early rounds are slower: random play makes long games).
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-python}
# The settings can be overridden from outside, which only the local smoke test does.
RUN=${RUN:-scratch_256}
ROUNDS=${ROUNDS:-500}
WIDTH=${WIDTH:-256}
LAYERS=${LAYERS:-8}
GAMES=${GAMES:-96}
COMPARE_GAMES=${COMPARE_GAMES:-100}
DRIVE=${DRIVE:-/content/drive/MyDrive/agent_game}
WORKERS=${WORKERS:-$(( $(nproc) - 1 ))}
mkdir -p "$DRIVE/runs/$RUN" ml/runs

keep() {  # copy what is worth keeping to Drive
    for file in ml/runs/$RUN/metrics.jsonl ml/runs/$RUN/settings.json ml/runs/$RUN/latest.pt ml/runs/$RUN/best.pt \
        ml/runs/$RUN/round_0[0-9][05]0.pt; do
        if [ -f "$file" ] && [ "$file" -nt "$DRIVE/runs/$RUN/$(basename "$file")" ]; then cp "$file" "$DRIVE/runs/$RUN/"; fi
    done
}

# A new runtime: carry on from what an earlier one kept on Drive.
if [ ! -f ml/runs/$RUN/latest.pt ] && [ -f "$DRIVE/runs/$RUN/latest.pt" ]; then
    echo "=== carrying on from Drive ($(date +%H:%M))"
    mkdir -p ml/runs/$RUN && cp "$DRIVE/runs/$RUN/"* ml/runs/$RUN/
fi

if [ ! -f ml/runs/$RUN/latest.pt ]; then
    echo "=== $RUN: a fresh ${WIDTH}x${LAYERS} network ($(date +%H:%M))"
    $PY -m ml checkpoint new ml/runs/${RUN}_start.pt --set width=$WIDTH layers=$LAYERS --seed 0 || exit 1
    $PY -u -m ml train selfplay --run $RUN --init ml/runs/${RUN}_start.pt --seed 0 --rounds $ROUNDS \
        --games-per-round $GAMES --inference server --workers $WORKERS \
        --entropy-weight 0.01 --entropy-half-life 100 --entropy-floor 0.0005 \
        --evaluate-every 10 --final-evaluation no --stop-after 50 --overwrite || exit 1
    keep
fi
until grep -q "\"round\": $ROUNDS," ml/runs/$RUN/metrics.jsonl; do
    echo "=== $RUN (resuming, $(date +%H:%M))"
    $PY -u -m ml train selfplay --run $RUN --resume --stop-after 50 || exit 1
    keep
done
echo "=== $RUN finished training ($(date +%H:%M))"

if [ ! -f "$DRIVE/runs/compare_$RUN/report.txt" ]; then
    echo "=== comparison ($(date +%H:%M))"
    $PY -u -m ml compare --name $RUN --gpu-server --workers $WORKERS \
        --run "${WIDTH}x${LAYERS}, shrinking entropy/seed 0=ml/runs/$RUN" \
        --run "64x6, entropy 0.01/seed 0=ml/runs/deep_scratch" \
        --rounds 0 100 200 300 400 500 \
        --panel strategic_ai aggressive_ai ml/runs/deep_from_imitation/round_0400.pt \
        --games $COMPARE_GAMES || exit 1
    mkdir -p "$DRIVE/runs/compare_$RUN" && cp ml/runs/compare_$RUN/* "$DRIVE/runs/compare_$RUN/"
fi
echo "=== $RUN all done ($(date +%H:%M))"
