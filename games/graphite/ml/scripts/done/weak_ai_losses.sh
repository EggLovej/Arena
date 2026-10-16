#!/bin/sh
# Why does our strongest network still lose to the weak scripted AIs sometimes?
#
# deep_from_imitation round 500, three ways: picking moves by chance (as in training), always its top
# move (--greedy), and with move scoring (8 candidates). Each against aggressive_ai and careful_ai on
# 100 fresh maps from both seats (200 games per pairing). Every loss is saved to ml/runs/losses/,
# then each lost start is played again 50 times from each seat (ml rematch).
#
# Run:  ml/scripts/done/weak_ai_losses.sh
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
NET=ml/runs/deep_from_imitation/round_0500.pt
for flags in "--candidate $NET" "--candidate $NET --greedy" "--candidate value:$NET?candidates=8"; do
    echo "=== evaluate $flags"
    $PY -u -m ml evaluate $flags --opponents aggressive_ai careful_ai --generated-maps 100 --no-bundled-maps \
        --skip-generated 30 --save-losses-against aggressive_ai careful_ai || exit 1
done
echo "=== rematches"
for game in ml/runs/losses/*/*.json; do
    [ -f "$game" ] || continue
    echo "--- $game"
    $PY -u -m ml rematch "$game" --games 50 || exit 1
done
echo "=== weak AI losses finished"
