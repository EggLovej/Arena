# ml/ — a learned agent for the game

Three documents go with the code:

- [research_tree.md](research_tree.md): everything tried, by topic, with how
  far each branch went and why it stopped. **Start here.**
- [plan.md](plan.md): status, next steps, how we work, decisions.
- [history.md](history.md): the build stages and every experiment, by date,
  with its numbers.

This page is the map of the code: where things are and how to run them.
All commands run from the game directory through one entry point:

```bash
.venv/bin/python -m ml                 # list the commands
.venv/bin/python -m ml <command> --help
```

## Commands

| Command | What it does | Example |
|---|---|---|
| `train` | Train a network: `imitation`, `selfplay`, `league`, or `kickstart` (a new network learns to play like a trained one) | `-m ml train selfplay --run demo --init random --monitor` |
| `monitor` | Live charts for a run (reads its log only) | `-m ml monitor --run demo` |
| `tensorboard` | The same curves in TensorBoard, in the browser: several runs in one chart, smoothing, log scale, x-axis by round or hours (reads the logs only) | `-m ml tensorboard --run free_scratch --run free_old_entropy` |
| `lineage` | One set of curves for a network trained over several runs, with linked ratings | `-m ml lineage ml/runs/shaping_entropy/round_0150.pt`, then `-m ml monitor --run lineage_shaping_entropy` |
| `evaluate` | Play a player against opponents on fresh maps; games lost to `aggressive_ai` are saved to `ml/runs/losses/` | `-m ml evaluate --candidate ml/runs/latest.pt --skip-generated 30` |
| `compare` | Rate the saved rounds of several runs on one scale (live board while it plays) | `-m ml compare --name a_vs_b --run "a/seed 0=ml/runs/a" --run "b/seed 0=ml/runs/b" --rounds 0 50 100` |
| `behaviour` | How networks play: where sends go, how far units are from the front, how decided the amounts are | `-m ml behaviour ml/runs/ab_plain/round_0400.pt` |
| `bot-ratings` | Rate the AIs in the game's menu (strategic_ai = 1000) for the home screen, into `ais/ratings.json` | `-m ml bot-ratings` |
| `sweep` | Train variations of the settings and rate them | `-m ml sweep --name hint --trainer selfplay --vary material_hint=0,0.5` |
| `summary` | Final numbers of several runs side by side, no games played | `-m ml summary capacity_current capacity_deeper` |
| `judge-games` | Judge loop, step 2: play many games, keep labelled positions | `-m ml judge-games --name round3 --seed 2` |
| `judge-train` | Judge loop, step 3: train the judge (win/draw/loss) | `-m ml judge-train --data round1 round2 --run judge3` |
| `judge-check` | Judge loop, step 4: the look-ahead with a new judge against the old one, Beast, Atlas | `-m ml judge-check --judge ml/runs/judge2/best.pt` |
| `policy-check` | A network with the look-ahead against Beast and Atlas, and on instinct against another | `-m ml policy-check --policy ml/runs/shaping_entropy/round_0150.pt --judge ml/runs/judge2/best.pt` |
| `beast` | Show or change who Beast AI (our strongest player, in the game menu) is | `-m ml beast set "lookahead:..." --why "..."` |
| `mistakes` | Where a network fails to copy `strategic_ai`: by part of `strategic_ai` and kind of mistake | `-m ml mistakes ml/runs/inputs_before_seed0/latest.pt` |
| `rematch` | Play a saved game's start again, 50 times from each seat, and count | `-m ml rematch ml/runs/losses/<candidate>/<game>.json` |
| `watch` | Watch a saved game in the game window (`--live` to replay it with the players) | `-m ml watch FILE --speed 3` |
| `benchmark` | How fast this machine trains: forward passes, a round of playing (CPU workers or GPU server), learning | `-m ml benchmark --sizes 64x6 128x8` |
| `colab` | Run commands on a Colab runtime: `build` writes a notebook that carries the code and checkpoints, `read` takes the output and any files back out of the saved notebook | `-m ml colab build --name bench --command "python -u -m ml benchmark"` |
| `rollout` | Play many games at once, report speed | `-m ml rollout --red untrained --blue strategic_ai --games 64` |
| `checkpoint` | Create or inspect a checkpoint (`--set amounts=free feature_set=v2`: free amounts, the v2 inputs) | `-m ml checkpoint new ml/runs/x.pt --set width=128 layers=8` |
| `maps` | Browse generated maps | `-m ml maps` |
| `show-state` | What the network sees in one position | `-m ml show-state --seed 3 --seconds 20` |
| `show-actions` | The moves it can choose from there; with `--network`, what a network would do (`free`: untrained, free amounts) | `-m ml show-actions --seed 3 --network free` |
| `show-model` | What a model makes of that position | `-m ml show-model` |

Wherever a player is named (evaluate, compare, trainers' opponents), it can
be an AI file from `ais/`, a checkpoint path, `value:PATH` for that
checkpoint picking the move whose resulting position it likes best, or
`lookahead:PATH?judge=JUDGE` for the two-move look-ahead (Beast's kind).

## Layout

Dependencies point downwards only (the full list of files is in `plan.md`):

```
encoding/, map_generator/   what the network sees and can do, the maps it plays on
models/                     the playing network, the judge, checkpoints
play/                       networks as players: agent, rollouts, inference, move scoring, look-ahead, Beast
training/                   trainers (imitation, selfplay, league, kickstart) and what they share
judge/                      the judge loop (games, training, checks)
evaluation/                 evaluate, compare, lineage, behaviour, sweeps, saved games
scripts/                    long jobs as shell scripts; finished ones in scripts/done/
tests/                      one test file per module
runs/                       one folder per run (settings, metrics, every 50th round); logs/ for job logs
```

`runs/latest.pt` always points at the newest training round of the most
recently started run; the game's "Neural AI" plays it unless
`NEURAL_AI_CHECKPOINT` names another file. "Beast AI" plays whatever
`runs/beast.json` names (`ml beast`).

## Tests

```bash
.venv/bin/python -m unittest discover -s ml/tests -t .   # about 5 minutes
```
