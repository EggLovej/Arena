# Committed checkpoints

The networks the learned bots play, so a fresh checkout (and the Arena
tournament on GitHub Actions) plays them without any training runs on disk.
Everything else training writes stays in `ml/runs/`, which git ignores.

| File | Played by | Copied from | SHA-256 |
| --- | --- | --- | --- |
| `neural_ai.pt` | `ais/neural_ai.py`, and Atlas's forecasts in `ais/atlas_ai.py` | `ml/runs/deep_from_imitation/round_0500.pt` (what `ml/runs/latest.pt` held on 2026-10-16) | `ae4c1caaceb9840b0602cb5091a9c48bf5052b0579404f7d2fb58ba1704dc957` |
| `beast_ai.pt` | `ais/beast_ai.py`, through `beast.json` | `ml/runs/free2_gap0/round_0050.pt` (Beast since 2026-10-13) | `63a00ae15a99f39eb53ad340651d7ba9211543bfc498fd71ecf576ade683421e` |
| `beast.json` | `ml/play/beast.py` | `ml/runs/beast.json`, with the path pointing here | |

Which file a bot plays, first match wins:

- **Neural AI:** `NEURAL_AI_CHECKPOINT`, then `ml/runs/latest.pt`, then `checkpoints/neural_ai.pt`.
- **Beast AI:** `BEAST_AI_FILE`, then `ml/runs/beast.json`, then `checkpoints/beast.json`.
- **Atlas AI's forecasts:** `ATLAS_OPPONENT_CHECKPOINT`, then `NEURAL_AI_CHECKPOINT`, then
  `ml/runs/latest.pt`, then `checkpoints/neural_ai.pt`.

The Arena adapter (`arena_game.py`) sets `NEURAL_AI_CHECKPOINT` and `BEAST_AI_FILE` to the files
here unless they are already set, so leaderboard matches always play the committed networks.

To publish a new network, copy it over the file here and commit it. A network trained under a
send limit records it in its own history and plays under it, so nothing else needs to come along.
