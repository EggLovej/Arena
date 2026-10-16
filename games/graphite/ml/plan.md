# ML agent: implementation plan

A living document: status, next steps, how we work, decisions, open
questions. Three files go with it:

- `ml/research_tree.md`: the map. Everything tried, by topic, with how far
  each branch went and why it stopped. Start there.
- `ml/history.md`: the details. The build stages and every experiment in
  the order it was done, with its numbers. A section named in quotes below
  ("The judge", "Inputs", ...) is there.
- `ml/README.md`: the layout and the commands.

**Goal:** an agent that learns this game by playing it, keeps improving, and
can be trained on one laptop. Just as important: every piece should be small
enough to read and understand on its own.

**Status (2026-10-04):**
- **Best network:** `shaping_entropy` round 150 (64 wide, 6 layers;
  imitation of `strategic_ai`, then 1,050 rounds of self-play, the last 150
  with the entropy bonus at 0.001 instead of 0.01): 1487 on the linked scale
  where `strategic_ai` is 1000, about +100 over the 0.01 runs, which had
  stopped improving. Its whole road: `ml monitor --run lineage_shaping_entropy`.
- **Plateau again:** 300 more rounds with a shrinking bonus (`entropy_decay`)
  stayed flat (1404 → 1394): the step to 0.001 was a one-off.
- **Running on Colab** (A100, through `ml colab`): a 256×8 network from
  random weights, self-play only (`scratch_256`).
- **Best player overall: Beast**, the two-move look-ahead (`nash`) with
  `deep_from_imitation` round 500 proposing moves and `judge1` judging them
  (82.5% against the original Atlas). Better proposers have not beaten it
  (51–52.5%); its judge decides.
- **What did not help** (see the tree): a league, larger networks (so far),
  new inputs, learning from the look-ahead or move scoring, the judge as
  critic, production and front potentials.

## Next steps

1. ~~Keep training the best network with a shrinking entropy bonus~~: flat
   over 300 rounds. More rounds of the same will not help the 64×6.
2. **A new judge for Beast**, from games of the stronger networks: running
   (`ml/scripts/judge3.sh`).
3. **The larger network again**, now that self-play improves and Colab makes
   it cheap: 256×8 from random weights, self-play only, on Colab (started,
   `ml/colab/scratch_256.ipynb`).
4. ~~Shuffling between own bases~~: checked (2026-10-08), about 80% of
   those sends move units toward the enemy, relayed one route at a time.
5. **Acting once a second** (an ablation, 2026-10-05): limited to one send
   a second, the best network drops 376 points; 150 rounds of training
   under the limit win back about half (1159 against the control's 1364).
   It shuffles less and decides its amounts more firmly, but does not learn
   to hold back. **With the limit lifted again it beats the unlimited
   control (61%, and 60.5% against the control's round 300)**: the first
   thing past the plateau since the entropy step. One seed; to confirm.
   Plan: (1) more rounds under the limit, judged with it lifted: done,
   1485 by round 200 and flat after, about +80 over the control (1404);
   (2) your change: lift it step by step from round 200, 50 rounds each
   at gaps 9 to 5: done, the gain held (68% against the control at gap 5,
   54% against the start), shuffling creeps back (63% → 71%); (3) a second seed,
   and Beast, before it becomes Neural AI's network. "Acting once a
   second".
6. **A new feature set and free amounts** (your design, 2026-10-08): stage 1,
   the inputs (`ml show-state --features v2`), stage 2, the free-amount
   output (`ml show-actions --network free`), and stage 3, every trainer
   (`--set amounts=free feature_set=v2`; demo `ml/scripts/free_demo.sh`), are
   built. Stage 4, first look: `free_scratch`, self-play only from random
   weights for 100 rounds, then rated next to `deep_scratch` at rounds 0, 50
   and 100 (`ml/scripts/free_scratch.sh`, done 2026-10-09): at round 100,
   1122 against 1108, 76% against 61.5% vs strategic_ai, but 44% head to
   head (−1.2 SE). About even, one seed, and the entropy bonus differed too.
   Carried on to round 300 as the same run
   (`ml/scripts/free_scratch_more.sh`, done 2026-10-09 17:59): at round 300,
   1289 against 1193, 86% against 74% vs strategic_ai, and 63% head to head
   (+2.6 SE). The new design pulled ahead with more training (about 1300 on
   the Stage 10 scale, above `deep_scratch` round 500, below Neural AI's
   1357). Still rising, more slowly. One seed, different entropy bonus,
   1.3 times the time per round.
   Then the new design with the old bonus (0.01 throughout), to tell which of
   the two did it: `free_old_entropy` (`ml/scripts/free_old_entropy.sh`),
   paused after round 124 (2026-10-10) for the entropy's two parts (now
   logged: `entropy/of the route choice`, `entropy/of the amounts`).
   Done 2026-10-11: `free_lookahead` (`ml/scripts/free_lookahead.sh`),
   `free_scratch` again with lookahead 0.995 instead of 0.99, everything
   else the same: behind at rounds 100/200/300 (1006/1121/1218 against
   1223/1245/1316; 35.5% head to head at round 300), and no less shuffling
   (it waits more instead). 0.99 stays. "A longer lookahead from scratch".
   The ramp and the second seed wait. "A new feature set and free amounts".
   **The user's plan (2026-10-11), four steps:** (1) `free_scratch` trains
   on until it stops improving: blocks of 100 rounds, after each its last
   round against the round 100 earlier (200 games); two blocks under 55% in
   a row end it, round 1500 at the latest (`ml/scripts/free_scratch_plateau.sh`):
   done 2026-10-11 18:38, flat from round 600; against Neural AI's network
   about 42% at every block from 400 to 800, so levelled off below it.
   (2) Done 2026-10-12 03:00 (`ml/scripts/free_gap_ramp.sh`), from round 800:
   gap 10 to round 250 (flat from 150), then 9, 8, ... down to none. The end
   (`free_gap0` r50) beats the control 72.8%, Neural AI's network 60.8%, and
   plays Beast even (48.8%).
   (3) Changed by the user (2026-10-11 23:15): when the ramp ends, evaluate
   it (the strongest of its networks, then against Beast), then straight away
   the strongest without a send limit at lookahead 0.995 for 150 rounds, one
   arm, no control (`ml/scripts/free_after_ramp.sh`). Done 2026-10-12 06:06:
   no measurable change (48.5–51.8% against its start over 150 rounds;
   round 100 51.5% against Beast). 0.99 stays. At temperature 0.2, `free_gap0`
   r50 beat Beast 55.5% over 400 games (+2.2 SE): **the new Beast** (2026-10-12,
   `network:ml/runs/free_gap0/round_0050.pt?temperature=0.2`); rated 1440 against
   Neural AI's 1433 (48.5% against it over 100 games): about even. Then a break.
   (4) Planned 2026-10-12 (history: "A judge for the free network"): stage 1,
   the look-ahead for free networks, done; stage 2 (`ml/scripts/judge4_stage2.sh`):
   the look-ahead on `free_gap0` loses to Beast, 34.5% judged by the network
   itself, 32.8% by `judge1`. **Parked** (the user, 2026-10-12).
   (5) A second round of the ramp from `free_gap0` r50 (`ml/scripts/free_gap_ramp2.sh`,
   done 2026-10-13 05:36): gap 10 flat from the start (ended at round 100),
   the steps 47–59% against the start by chance. At temperature 0.2 the final
   one (`free2_gap0` r50) beat `free_gap0` r50 60.8% (+3.0 SE) and drew with
   the best by the yardstick (`free2_gap3` r50, 56.5% against `free_gap0`):
   **the new Beast** (2026-10-13,
   `network:ml/runs/free2_gap0/round_0050.pt?temperature=0.2`). History:
   "A second round of the send-limit ramp". Steps 1–3 in the
   history: "Training free_scratch until it stops improving".
7. Smaller: does the random start catch up (Stage 10, more seeds); a
   control arm at a constant entropy bonus; three-player games.

## How we work

- One stage at a time. A stage is finished when it has code, tests, and a
  small demo you can run and look at.
- Each stage below says **what** we build, **why** it exists, and the
  **concept** it teaches. If the why is not convincing, we stop and discuss
  before building.
- Predictions are written down *before* we measure them (see "Predictions").
  Being wrong about one is a result, not a failure.

## The big picture

```
            ┌────────────────────────────────────────────┐
            │                                            ▼
   play games (rollout)  ──►  trajectories  ──►  update the network
            ▲                                            │
            └──────────────── new weights ◄──────────────┘
```

Every learning method here is this loop. They differ only in **who plays** and
**what the network is told to aim for**:

| Method | Who plays | The network is trained to... |
|---|---|---|
| Imitation | a scripted teacher (`strategic_ai`) | copy the teacher's move in each position |
| Self-play RL (PPO) | the network against itself and a league | make moves that led to winning more likely |
| Search + distil (optional) | the network, helped by lookahead | copy what the lookahead preferred |

## Two starting points, one pipeline

The central design rule: **a checkpoint is the only thing trainers exchange.**
A checkpoint is weights + the config that built the network + a note on where
it came from. Every trainer takes `--init random` or `--init <checkpoint>` and
writes checkpoints. Trainers never import each other.

That makes the two routes you want to compare the same code with different
arguments:

```bash
# Route A: from scratch
python -m ml train selfplay  --run scratch --init random

# Route B: learn from a teacher first, then improve
python -m ml train imitation --run bc --teacher strategic_ai
python -m ml train selfplay  --run bc_then_selfplay --init ml/runs/bc/latest.pt

# Same opponents, same maps, one rating scale
python -m ml compare --name a_vs_b --run "scratch/seed 0=ml/runs/scratch" \
    --run "imitation/seed 0=ml/runs/bc_then_selfplay" --rounds 0 50 100
```

## Layout

Every command goes through one entry point, run from the game directory:
`.venv/bin/python -m ml <command>` (`-m ml` alone lists them). See
`ml/README.md` for the commands.

```
ml/
  __main__.py      the entry point: python -m ml <command>
  README.md        layout and commands at a glance
  plan.md          this file: status, next steps, decisions
  research_tree.md everything tried, by topic: how far, and why it stopped
  history.md       the build stages and every experiment, by date, with numbers
  map_generator/   making maps
    generator.py         random map generator and its parameters
    viewer.py            window for browsing generated maps
  encoding/        what the network sees and what it can choose
    state_encoder.py     GameView -> arrays (base, pair, route, global features)
    forecast.py          what fleets in flight will do to a base; capture, hold, spare amounts
    action_encoder.py    choice <-> SendAction, legal choices and their amounts
    show_state.py, show_actions.py   print one position's encoding or choices
  models/          the networks and their files
    interface.py         what every model takes and returns
    policy.py            output -> choices, probabilities
    amounts.py           free amounts: curves over every whole number of units
    graph_transformer.py the playing network (optional capture output)
    judge.py             the judge: who wins from a position (separate value network)
    checkpoint.py        save / load / --init handling, ml checkpoint
    show_model.py        print what a model makes of one position
  play/            turning networks into players
    agent.py             network -> choose_action(state)
    rollout.py           many games at once, records decisions (and where sends end up)
    inference.py         who runs the networks during rollouts: each worker, or one GPU server
    value_player.py      move scoring: judge the position after each candidate
    lookahead.py         two-move look-ahead: our candidates against theirs (Beast)
    players.py           value: and lookahead: player names
    beast.py             which player Beast is (ml beast)
  training/        the trainers and what they share
    train.py             command: start any trainer
    dataset.py           recorded decisions as batches
    runlog.py            a run's folder: metrics, checkpoints
    monitor.py           live charts for a run
    board.py             the same curves in TensorBoard, several runs at once (ml tensorboard)
    imitation.py         trainer: copy a teacher
    selfplay.py          trainer: self-play (PPO), entropy schedule, potentials
    league.py            trainer: self-play against past champions and scripted AIs
    kickstart.py         trainer: a new network learns to play like a trained one
  judge/           the judge loop
    games.py, train.py, check.py      ml judge-games / judge-train / judge-check
    policy_check.py      a network with the look-ahead against Beast and Atlas (ml policy-check)
    progress.py          live progress blocks and lines
  evaluation/      measuring players
    evaluate.py          win rates against opponents on fresh maps; saves lost games
    compare.py           one rating scale for the rounds of several runs
    lineage.py           one set of curves for a network trained over several runs
    behaviour.py         how a network plays: where sends go, how decided it is
    bot_ratings.py       ratings of the menu's AIs for the home screen (ais/ratings.json)
    summary.py, sweep.py, mistakes.py  side-by-side numbers, settings sweeps, copy mistakes
    saved_games.py, rematch.py, watch.py   saved games: play again, watch
  scripts/         long jobs; finished ones in scripts/done/
  tests/           one test file per module, plus support.py for shared helpers
  runs/            one folder per run (every 50th round kept), logs/ for the jobs' logs
ais/neural_ai.py   the network in the game (Neural AI)
ais/beast_ai.py    our strongest player in the game (Beast AI)
```

Dependencies only point downwards: encoding and models know nothing of
playing; play knows nothing of training; training and evaluation use play.

Nothing in `ml/` changes `game.py`, `game_types.py`, or the existing AIs.

## Predictions (written before measuring)

| # | Prediction | Test in | Result |
|---|---|---|---|
| 1 | Games between untrained networks mostly hit the 180 s limit without a winner | Stage 5 | **Wrong.** 58% of 76 games had a winner. On the bundled maps all 16 did; on generated maps 47% did. Random play ends games more often than expected, so a random start is less hopeless than assumed. |
| 2 | Route B is ahead early; whether Route A catches up is unknown | Stage 10 | **Ahead early: right.** On the deep network it was also ahead at round 500 (+94 points, 65.5% head to head), but whether the random start catches up later is still open. One seed. |
| 3 | A pure network will not beat a rollout-search controller without Stage 11 | Stage 10–11 | **Right so far.** The overnight model (round 500) lost 4–12 to Atlas. Our own search beat the plain network 21–9. |
| 4 | The route features raise agreement on `strategic_ai`'s sends clearly above 71%, by more than the two seeds differ; the production shares add nothing measurable in a 1v1, since they follow exactly from two global features already there | Inputs | **Wrong about the routes:** 72.4% and 67.7% against 73.5% and 66.2% without them, inside the noise. The production shares came out 2.5–4.4 points lower on both seeds; probably noise, not proven either way. |
| 5 | Limited to one send a second without retraining, the best network wins under 20% of 200 games against itself unlimited, and drops below 50% against strategic_ai: 88% of its sends would be blocked, and the one that gets through each second is usually a shuffle between its own bases | Acting once a second | **Right.** 19.5% of 200 games against itself unlimited, 43.5% against strategic_ai; 980 against 1356. |
| 6 | Trained under the limit for 150 rounds, it wins back at least half of that rating loss: larger sends (at least 3x the units per send) and under 50% of them to its own bases. It still loses head to head to the unlimited control | Acting once a second | **Mostly wrong.** It won back 179 of 376 points (48%, just short of half). Its sends carry 3.5 times the units, but already untrained, so that comes from the limit, not from learning. Sends to own bases 65%, not under 50%. Still loses to the control: 27%. |

## Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-09-30 | PyTorch | Most widely documented; MPS gives ~10x on batched inference here |
| 2026-09-30 | Checkpoints are the only interface between trainers | Makes random-start and imitation-start the same code path |
| 2026-09-30 | Train on generated maps only; bundled maps are validation | Gives an honest generalisation number |
| 2026-09-30 | No changes to the engine or existing AIs | `ml/` stays removable |
| 2026-09-30 | Generator gets an `asymmetry` setting from the start; fairness is handled by seat swapping | 7 of 11 bundled maps are asymmetric, so mirrored-only training would not match validation |
| 2026-09-30 | Stage 1 is 1v1 only | Your call; three-player maps are added later |
| 2026-09-30 | No balance filter on generated maps; the balance probe is a diagnostic only | Deterministic self-play gives one fixed result per map. With noise added it works as a number, but it measures one agent's play and would reject 5 of 11 bundled maps |
| 2026-09-30 | Value limits: production 0–10, starting units 0–250, neutral units 1–250 | Your numbers |
| 2026-09-30 | Inside the limits, values are drawn around a typical value with a spread, not evenly | An even draw would make the average neutral base cost 125 against a typical start of 25, so most maps would stall |
| 2026-09-30 | Travel time follows route length | The picture then matches the numbers; asymmetry in positions becomes asymmetry in travel times for free |
| 2026-09-30 | Encoder output is NumPy, not PyTorch | The encoder stays usable without the network and is easy to test and print |
| 2026-09-30 | Route features are computed once per match; only base and global features are rebuilt each tick | Routes never change, and fleets are summarised at their target base |
| 2026-09-30 | Amounts are eighths of the source's units, plus "just enough to capture" | Eighths reproduce 93% of `strategic_ai`'s sends exactly against 54% for quarters, at the cost of four more options |
| 2026-09-30 | The legal-choice list carries the unit count of every option | Decoding is a lookup, so the mask and the move can never disagree |
| 2026-09-30 | Models live in `ml/models/` behind one interface and are built by name | Architectures can be swapped without touching trainers or the agent |
| 2026-09-30 | Architectures return plain scores; legality is applied in the shared base class | No architecture can get masking wrong, and all of them are compared on equal terms |
| 2026-09-30 | Of the legal-choice table, the network is shown only `can_send_just_enough` per route | The eight fraction options are legal whenever the route is, which the network can already see; only "just enough" carries new information. The cost of "just enough" is a candidate extra input to test later |
| 2026-09-30 | A checkpoint records its history, one entry per step that produced it | The random-start and imitation-start routes stay distinguishable from the file alone |
| 2026-09-30 | Loading refuses a checkpoint whose feature lists differ from the current encoders | A model fed different inputs than it was trained on would fail silently otherwise |
| 2026-09-30 | Generated maps from seed 900,000 upwards are for evaluation only | Training must never see the maps it is measured on |
| 2026-09-30 | Maps saved from the viewer (`generated_*`) are left out of the bundled evaluation set | They are not hand-made, and saving one should not change the ruler |
| 2026-09-30 | Evaluation draws the network's moves by chance, as in training; `--greedy` is an option | Greedy play of an undecided network is degenerate: it repeats one move |
| 2026-09-30 | Save a checkpoint every training round | 0.5 MB each, so 1,000 rounds cost 0.5 GB; makes every point on the curve replayable |
| 2026-09-30 | Rollout workers run their own copy of the model on the CPU | Simple, and faster than sharing the GPU between workers at these batch sizes |
| 2026-09-30 | Positions where waiting is the only legal choice are not recorded | There is nothing to decide, so nothing to learn from |
| 2026-09-30 | A timeout is a real ending, not an interruption | The network sees the clock, so it can anticipate the draw; no special handling needed in training |
| 2026-09-30 | Training batches come from one map size at a time | No padding needed; each size is drawn as often as its share of the decisions |
| 2026-09-30 | At most 300 decisions are kept per seat and game | Long stalemates would otherwise supply most of the examples |
| 2026-09-30 | Validation is split off by game, not by decision | Neighbouring ticks of one game are nearly identical, so a split by decision would flatter the result |
| 2026-09-30 | The monitor is a separate program that only reads the run's log | It can be opened, closed, or crash without touching training |
| 2026-09-30 | One chart per group of metrics, never two scales on one chart | Two scales on one chart invite false comparisons |
| 2026-09-30 | The value output learns the actual result, not a bootstrapped estimate | Simple, and its range of -1 to +1 matches the output; the noisier target is affordable with this many games |
| 2026-09-30 | The material rule of thumb enters as part of the expectation, and fades out | Written as a reward it would be the same thing mathematically, but this way the value output always means "chance of winning" |
| 2026-09-30 | Self-play keeps `decision_interval` at 1 | The imitation model and the game's Neural AI decide every tick; training at another rhythm would change how often it sends |
| 2026-09-30 | Self-play updates use at most 200 decisions per seat and game | Going over every decision four times would take longer than playing the games |
| 2026-09-30 | In value-only rounds the moves are held in place by a penalty on any shift | The value output shares most of the network with the moves. Correcting it alone moved the imitation model's policy far more than any normal round is allowed to |
| 2026-09-30 | The step size adjusts itself to keep the policy change near a target | An undecided policy barely moves at a step size that throws a confident one around; one fixed value cannot suit both starting points |
| 2026-09-30 | The stop on policy change is checked after every batch | A confident policy passed the limit tenfold within a single pass |
| 2026-09-30 | The league reuses the self-play loop and only replaces where opponents come from | One loop to understand and to fix; the two trainers cannot drift apart |
| 2026-09-30 | Promotion needs 55% over 128 games on fresh maps | A network no better than the champion gets there by luck about one time in eight; a stricter gate would cost more games per check |
| 2026-09-30 | A failed promotion match does not reset the network | It keeps training and tries again five rounds later |
| 2026-09-30 | Ratings are refitted from all results each time, not updated match by match | Every match then informs every rating, and the order of matches does not matter |
| 2026-09-30 | Yes/no settings are given as `yes` or `no` on the command line | Before this, `--final-evaluation False` was read as "yes" |
| 2026-09-30 | Headline comparisons are made on fresh maps (`--skip-generated 30`), with as many games as practical | A model chosen for doing well on a set of maps looks better on those maps than it is |
| 2026-09-30 | Every learning batch has the same size, and the route list is padded to a few fixed lengths | On the Apple GPU each new shape costs memory for good; a long run otherwise fills the machine |
| 2026-10-01 | Search thinks only every third tick and only when the network is not sure it should wait | Thinking on every tick took about 430 ms per tick, which makes one game several minutes; this costs about 30 ms per tick |
| 2026-10-01 | Distillation reuses the imitation trainer with the search agent as the teacher | No new training code; one round of distillation is one command |
| 2026-10-01 | The distillation loop is its own trainer, built from the imitation trainer's pieces | The loop needs things a single imitation run has not: a teacher that changes every iteration, remembered decisions, checks against the previous iteration |
| 2026-10-01 | The league's memory is saved inside each round's checkpoint, champions by reference to their round files | Weights and league state can never be out of step after a crash, and no weights are stored twice |
| 2026-10-01 | Move scoring is used for playing, not as a distillation teacher | Ten iterations of copying its choices took the network from 1195 to 968; with the policy as a filter it beats the plain network 64% at play time |
| 2026-10-01 | Plain self-play, not the league, for the long run | From the overnight plateau, 50 rounds of either changed nothing measurable (2 seeds each, 400 games per rating); self-play is simpler and was tuned |
| 2026-10-01 | A deeper route head is an optional setting, off by default | Saved networks keep loading unchanged, and the probe can test it next to width and depth |
| 2026-10-01 | Code grouped by what it is for (encoding, models, play, training, evaluation), one `python -m ml` entry point | 25 files side by side were hard to find one's way in; one entry point means one command to remember |
| 2026-10-01 | For long runs, use the final model rather than "best" | In the overnight run the final round beat the round the quick check chose, on 360 fresh games |
| 2026-10-01 | Keep the default settings for long runs | Tuning found no better setting from either starting point, and a higher step-size ceiling was clearly worse |
| 2026-10-01 | Long runs restart in a fresh process every 50 rounds, resuming exactly | Keeps the remaining slow memory growth harmless without further changes to the model |
| 2026-10-02 | Features are only added at the end of a list; a model reads the first N of each, as recorded in its settings | New inputs can be tried without breaking every saved network, and old and new networks can still play each other |
| 2026-10-02 | Opponents in global features are slots ranked by strength, 0 when absent | Works for 1v1 and three players alike without a separate "absent" input |
| 2026-10-02 | Beast is always our strongest player, moved only after a look-ahead check against it (`ml beast set`) | A better network on instinct has twice not made a better look-ahead player |
| 2026-10-03 | Amount options are only appended ("hold", "surplus"); older networks never pick options they do not know | Every saved network keeps loading; checkpoints record the option list, so removing one would break them |
| 2026-10-03 | Rollouts keep the networks on the CPU workers; the GPU server is an option (`--inference server`) | On this Mac the server is no faster at any network size tried |
| 2026-10-03 | No judge as critic in self-play | Frozen, the network exploited it; learning along, it made the network worse |
| 2026-10-04 | Hand-made rules of thumb enter as potentials (credit for changes), never as event bonuses | Bonuses for events can be collected again and again; potentials cannot |
| 2026-10-04 | Entropy bonus 0.001 from round 900 on, shrinking with a floor (`--entropy-half-life`, `--entropy-floor`) | At 0.01 the amounts became nearly random and self-play stopped improving; at 0.001 it improved again (67% head to head) |
| 2026-10-04 | Runs keep every 50th round (plus 0, the last, best and latest); logs in `ml/runs/logs/`, finished scripts in `ml/scripts/done/` | 6.1 GB of checkpoints were mostly never looked at again |
| 2026-10-04 | Removed: the distillation trainer, the search player, teaching from the look-ahead, the judge as critic | Each tried and dead (see `research_tree.md`); their results stay in `history.md` |
| 2026-10-05 | A send limit is a property of a seat (`min_send_gap`), and measurements play a run's networks under the limit the run trained under | The engine stays untouched; a limited network is never measured by rules it did not learn |
| 2026-10-08 | Feature set v2 is appended to the feature lists and chosen by name (`FEATURE_SETS`) | Appending keeps every saved network loading; a set by name lets a network read columns that are not the first N |
| 2026-10-08 | A network built with default settings reads only the columns from before v2 (`COLUMNS_BEFORE_V2`) | Otherwise every new default network would silently read both the log and the plain copy of each count |
| 2026-10-08 | Free amounts: per route a blend of 3 logistic curves over whole units, each centred on fraction × most + share × needed + offset | The network chooses proportions; the exact numbers (units at the source, cost of a capture) come from the game, so hitting "exactly enough" takes no arithmetic inside the network |
| 2026-10-08 | Free-amount curves start on "just enough", "half" and "everything" | The most used of the old options, so an untrained network starts from sensible amounts |
| 2026-10-09 | Recorded games keep four numbers per route (most, needed, hold, surplus), from which every amount option is worked out | A free student can then learn an option teacher's probabilities per number of units, with options that send the same units added together |
| 2026-10-09 | The amount entropy works on five fixed grid widths, each route on the smallest that holds it, and route lists outside the network are padded as well | Grid width and route count changed with every batch, and on the Apple GPU each new shape costs memory for good: `free_scratch` grew to 31 GB by round 28 |
| 2026-10-09 | Trainers give the GPU's cache of freed buffers back once per round or epoch | That cache, not compiled shapes, was most of the memory that built up (3.4 GB down to 0.5 GB when released); once per round costs almost nothing, every step would double the time |
| 2026-10-09 | The GPU cache is released only once the GPU holds more than 4 GB | Releasing every round made learning about 30% slower (buffers allocated afresh each round) and did not lower the peak during learning (about 3.3 GB) |
| 2026-10-09 | Long runs no longer need to restart every 50 rounds | With the cache released each round, `free_scratch` stayed at 1.6–2.1 GB over rounds 135–165, growing about 2 MB a round |
| 2026-10-09 | Rollouts still record every input column | Measured 640 MB per 96-game round; a v2 network reads 34 of 41 base columns, so recording only those would save about 15% |
| 2026-10-10 | Neural AI in the game draws its moves at temperature 0.2 (`ais/neural_ai.py`), first 0.8; training and measurements stay at 1 | The user's choice: often its favourite (about 70%), not always. 0.8 changed too little (33%, shuffled as much). 0.2 beats drawing at 1 by 63.5% to 36.5%, is even with always the favourite (51%), and shuffles about as little; rated 1420 against 1332 |
| 2026-10-10 | The game runner takes a temperature per network (`temperatures` in `play_games` and `RolloutPool.play`); `ml bot-ratings` rates Neural AI at its temperature | Matches recorded against Neural AI drawing by chance now count for that player (`neural_ai drawing by chance`, which also plays the quick players), so they still join Beast and the look-ahead to the scale |
| 2026-10-10 | TensorBoard for curves next to the pygame monitor (`ml tensorboard`), fed by copying the runs' metrics files | Smoothing, log scale and several runs in one chart come built in; the trainers and run folders stay as they are, and every old run can be shown |
| 2026-10-05 | The home screen shows each AI's estimated rating from `ais/ratings.json` (`ml bot-ratings`): the quick players play fresh, Atlas, Atlas Clean and Beast come from recorded matches | Atlas needs about 3 minutes per game against a network; the recorded matches already connect them to the scale |

## Open questions

- **Colab for training?** There is no direct link from this session to a
  Colab runtime. The workable route is through files: the dataset and the
  code go to Google Drive (or are uploaded), a notebook trains and writes
  the checkpoint back, and it comes down to `ml/runs/`. Remote shells into
  Colab (SSH tunnels) run against Colab's usage rules. Worth it only when
  training, not game generation, becomes the slow part.

- **Atlas without copies of its opponents** is about as strong as our
  overnight network (see "Atlas as a yardstick"). Its own forecaster for
  opponents (ten hand-made styles) is the weak part; a trained network
  forecaster that Atlas owns, not the opponent's, would be the fair upgrade.
- **What limits copying the sends?** The capacity probe says it is not size.
  Candidates: what the network is shown (for example the cost of "just
  enough" on every route, also when it cannot yet be afforded, which might
  help it learn to save up), or the amount of data.
- What mix of mirrored and asymmetric maps to train on, and how lopsided is
  too lopsided? Current guess: half mirrored, the rest up to asymmetry 0.3.
  The probe cannot settle this: between two identical players even asymmetry
  0.02 decides 25 of 60 maps. Judge by eye in the viewer for now.
- Seeds 10000–10011 were already inspected in earlier research, so final
  holdout maps need new seeds.
- When our networks beat every scripted AI almost every time, the scripted
  AIs stop pinning down the rating scale. Differences between networks stay
  well measured (they play each other), but the absolute numbers drift.
- Rollout speed is about 10,000 decisions/s and does not rise past six
  processes. If it becomes the limit, the next step is one shared model on
  the GPU answering all workers (about 35 µs per decision against about 180
  µs on a CPU thread), at the price of more plumbing.
- A single training process grows by about 40 MB per round (new
  combinations of map size and route count). Long runs restart every 50
  rounds (`--stop-after`, then `--resume`), which works for self-play,
  league, and distillation. Padding map sizes into a few buckets would stop
  it at the root.
- `ml/runs/latest.pt`, which the game's Neural AI plays, follows the most
  recently started run, even an untrained one. Point `NEURAL_AI_CHECKPOINT`
  at a specific file to pin it. Long scripts put it back when they end.
- Production of exactly 0 is allowed by the limits but never drawn. If bases
  that produce nothing are wanted, add an explicit chance for them.
- A saved generated map lands in `maps/`. `ml/evaluation/evaluate.py` now ignores it, but
  the game's own `evaluate.py` still includes it.
