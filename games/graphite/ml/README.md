# ml/ — a learned agent for the game

The networks behind Instinct (`ais/neural_ai.py`) and Apex (`ais/beast_ai.py`):
what they see, how they are trained and measured, and how they play. All
commands run from the game directory through one entry point:

```bash
.venv/bin/python -m ml                 # list the commands
.venv/bin/python -m ml <command> --help
```

## Train, measure, promote

The main line, as the committed networks were made:

```bash
# 1. Copy a scripted teacher (strategic_ai), 2. then get better by playing itself
.venv/bin/python -m ml train imitation --run teacher
.venv/bin/python -m ml train selfplay --run student --init ml/runs/teacher/best.pt
# 3. Rate its saved rounds against the scripted bots, on one scale with other runs
.venv/bin/python -m ml compare --name student --run "student/seed 0=ml/runs/student" --rounds 0 50 100
# 4. Play the best round against the current Apex, on maps no trainer has looked at
.venv/bin/python -m ml evaluate --candidate "network:ml/runs/student/round_0100.pt?temperature=0.2" \
    --opponents beast_ai --skip-generated 30
# 5. Promote it: copy it over the network a bot plays, and update the table in checkpoints/README.md
cp ml/runs/student/round_0100.pt checkpoints/beast_ai.pt
```

Two tests then check the promoted network: `python -m pytest -q ml/tests/test_numpy_model.py`
(the website's copy of the network scores it as PyTorch does) and, from the repository root,
`python -m pytest -q platform/tests/test_graphite.py -k committed_networks` (both bots play their
networks through the Arena). Instinct's network also makes Atlas's forecasts (`ais/atlas_ai.py`).

In the game window the bots play your newest work first: Instinct plays `ml/runs/latest.pt`
(the newest round of the most recently started run) unless `NEURAL_AI_CHECKPOINT` names
another file, and Apex plays whatever `ml/runs/beast.json` names (`ml beast set`). Without
those, and always in the Arena, they play the files in `checkpoints/`.

How the committed networks were made is in their own history (`ml checkpoint show FILE`):

- `checkpoints/neural_ai.pt`: 20 epochs imitating `strategic_ai` (600 games), then 500
  self-play rounds of 96 games.
- `checkpoints/beast_ai.pt`: free amounts and the v2 inputs (`ml checkpoint new FILE --set
  amounts=free feature_set=v2`), 800 self-play rounds from scratch, then twice a send limit
  lowered step by step: `--min-send-gap 10` (250 rounds, the second time 100), then 9, 8, … 2
  and none, 50 rounds each. Apex plays it at temperature 0.2.

## Playing without PyTorch

The website plays the bots in the visitor's browser, where PyTorch cannot run, so
`play/numpy_model.py` repeats the network's forward pass in numpy and `play/numpy_agent.py`
plays through it. A change to the network (`models/graph_transformer.py`,
`models/interface.py`, `choose` in `models/policy.py`, or `models/amounts.py`) has to be
made in `numpy_model.py` too; `tests/test_numpy_model.py` fails until it is.

## Rules the code keeps

- Input features and amount options are only ever added, at the end of their lists. A
  checkpoint carries the lists it was trained with, and loading refuses one whose lists
  have changed in any other way.
- Generated maps from seed 900,000 on are for measuring; trainers draw their maps below it.
  Trainers check their progress on the first few (so `evaluate --skip-generated 30` judges
  on maps none of them has looked at), and `compare` plays from 900,300 on.
- A checkpoint records the send limit (`--min-send-gap`) its run trained under. Instinct and
  `compare` play it under that limit.

## Players

Wherever a player is named (evaluate, compare, a trainer's opponents, `ml beast`), it can be
(`play/players.py` reads them):

| Name | Player |
|---|---|
| `strategic_ai` | an AI file in `ais/` |
| `ml/runs/x/best.pt` | a checkpoint, drawing its moves by its own odds |
| `untrained`, `untrained:7` | a network with random weights (from seed 7) |
| `network:PATH?temperature=0.2` | a checkpoint drawing at that temperature (lower: its likelier moves more often); Apex is `network:checkpoints/beast_ai.pt?temperature=0.2` |

## Commands

| Command | What it does | Example |
|---|---|---|
| `train` | Train a network: `imitation` or `selfplay` | `-m ml train selfplay --run demo --init random` |
| `tensorboard` | Training curves in TensorBoard, in the browser: several runs in one chart, smoothing, log scale, x-axis by round or hours (reads the runs' logs only) | `-m ml tensorboard --run free_scratch --run free_old_entropy` |
| `evaluate` | Play a player against opponents on fresh maps | `-m ml evaluate --candidate ml/runs/latest.pt --skip-generated 30` |
| `compare` | Rate the saved rounds of several runs on one scale (live board while it plays) | `-m ml compare --name a_vs_b --run "a/seed 0=ml/runs/a" --run "b/seed 0=ml/runs/b" --rounds 0 50 100` |
| `beast` | Show or change who Beast AI (Apex, our strongest player) is | `-m ml beast set "network:..." --why "..."` |
| `rollout` | Play many games at once, report speed | `-m ml rollout --red untrained --blue strategic_ai --games 64` |
| `checkpoint` | Create or inspect a checkpoint (`--set amounts=free feature_set=v2`: free amounts, the v2 inputs) | `-m ml checkpoint new ml/runs/x.pt --set width=128 layers=8` |
| `maps` | Browse generated maps | `-m ml maps` |

## Layout

Dependencies point downwards only:

```
encoding/, map_generator/   what the network sees and can do, the maps it plays on
models/                     the playing network and its checkpoints
play/                       networks as players: names, agents (PyTorch and numpy), rollouts, inference, Beast
training/                   trainers (imitation, selfplay), what they share, and their charts
evaluation/                 evaluate and compare
tests/                      one test file per module
runs/                       one folder per run (settings, metrics, every 50th round); git ignores it
```

## Tests

```bash
python -m pytest -q ml/tests   # as CI runs them
```
