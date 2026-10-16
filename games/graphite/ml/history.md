# ML agent: history of what was built and measured

Everything that used to make up most of `plan.md`, moved here unchanged on
2026-10-04: the build stages, then every experiment in the order it was
done, with its numbers. `plan.md` keeps the status, next steps, decisions,
and open questions; `research_tree.md` is the map of all of it by topic.

## Stages

### Stage 0 — Setup ✅
- [x] `ml/` folder and this plan
- [x] `torch` and `numpy` installed in `../.venv`, pinned in `ml/requirements.txt`
- [x] Inference speed of a stand-in network measured (see "Measurements")

### Stage 1 — Map generator (`map_generator/`) ✅
- **What:** seeded random 1v1 maps (three-player maps come later). Built in
  two steps: first a perfectly mirrored map, then an `asymmetry` setting
  (0 = mirrored) that perturbs it. Every tunable number is a field of
  `MapGenParams` at the top of `map_generator/generator.py`.
- **Try it:** `../.venv/bin/python -m ml maps` (add `--seed 42` or
  `--asymmetry 0.3`). Next / Previous or the arrow keys browse seeds, Up / Down
  fix the asymmetry level, S saves the map into `maps/` so it can be played.
- **Tests:** `../.venv/bin/python -m unittest ml.tests.test_map_generator`
- **Why:** with only a handful of maps the network would memorise them.
  Unlimited maps force it to learn the game. The bundled maps then become a
  test it has never trained on. Most bundled maps are *not* symmetric (see
  "Measurements"), so training on mirrored maps alone would not match the test.
- **Fairness comes from swapping seats, not from the map.** Every map is
  played once from each seat, so an unfair map cannot flatter one agent.
- **Lopsidedness is controlled by construction, not by a filter.** At
  asymmetry 0 a map is fair by symmetry, with nothing to measure. The setting
  then moves away from that in bounded steps. We do not reject maps by a
  measured balance score: the only score available today would reject 5 of
  the 11 bundled maps (see "Measurements"), which would bring back the
  train/test mismatch.
- **Concept:** train / validation / holdout splits, why generalisation is
  measured rather than assumed, and separating "the map was unfair" from "the
  agent played better".
- [x] Mirrored generator producing valid `MapDefinition`s (passes `validate_map`)
- [x] `asymmetry` setting: moves bases (which changes travel times), jitters
      units and production, drops routes
- [x] Parameters defined in one place (`MapGenParams`)
- [x] Map viewer window with Next / Previous / Save
- [x] Tests (15), including exact mirroring at asymmetry 0, limits respected,
      no crossing routes, and scripted AIs playing generated maps
- [x] Balance probe run once as a check (results under "Measurements"). Not
      kept as a module: it confirmed mirrored maps have no seat bias, but it is
      far too sensitive to say how much asymmetry is too much.
- [x] Your review: maps look good (2026-09-30)
- [ ] Later: three-player maps
- [ ] Later, if wanted: different starting setups per player (one fortress
      against several small bases, as in `siege`)
- [ ] Later (Stage 9): the trained network gives a better balance estimate for
      free, from its own seat win rates and its "who is winning" output at
      tick 0. Could be used to train less on maps the seat decides.
- **Done when:** scripted AIs play full matches on generated maps at several
  asymmetry settings (met), the probe shows no seat bias at asymmetry 0 and
  drifts away as the setting increases (met), and you are happy with how the
  maps look.

### Stage 2 — State encoder (`encoding/state_encoder.py`) ✅
- **What:** turn a `GameView` into fixed-shape arrays: one row per base
  (14 numbers), a base-by-base table of route features (3 numbers per pair),
  and 4 global numbers. The column names are listed at the top of the file.
- **Try it:** `../.venv/bin/python -m ml show-state --seed 3` prints
  the tables next to the game's own values; add `--seconds 20` for a position
  with fleets in flight. The seed matches the map viewer.
- **Tests:** `../.venv/bin/python -m unittest ml.tests.test_forecast ml.tests.test_state_encoder`
- **Why:** networks consume numbers, not dataclasses. The choices here decide
  what is easy or hard to learn. Three deliberate ones: everything is relative
  to the acting player (me / neutral / enemy), no coordinates or base IDs are
  included, and the result of fleets already in flight is computed exactly
  and handed over instead of being left for the network to work out.
- **Concept:** representation, symmetry, and feature scaling (unit counts are
  unbounded, so they go in as logs and shares).
- [x] Encoder with padding and a mask for maps of different sizes
- [x] Forecast of fleets in flight (`forecast.py`), matching the real `Game`
      on 500+ checks from generated-map matches
- [x] Symmetry tests: renaming and reordering bases, moving them on screen,
      or swapping colours gives the same encoding; on a mirrored map both
      seats see the same start
- [x] Demo script (`show_state.py`)
- [x] Your review of the feature list (2026-09-30)
- **Known simplifications** (revisit if the network struggles):
  - Fleets are summed per target base. The network sees how much is coming,
    how soon the first lands, and the final outcome, but not each wave or
    where it came from.
  - "Future" means a different moment for each base: whenever its own last
    incoming fleet lands.
  - 1v1 only: every other player is "the enemy".
  - The `time` feature assumes a 180 s match.
- **Done when:** the symmetry tests pass on all bundled and generated maps
  (met) and you are happy with the feature list.

### Stage 3 — Action encoder (`encoding/action_encoder.py`) ✅
- **What:** the network chooses "wait" or a route (source -> target), then an
  amount from nine options: "just enough to capture" or an eighth, two
  eighths, ... all of the units at the source. This module lists the legal
  choices in a position, with the units each would send, and converts a choice
  into a `SendAction`. It also maps a teacher's move onto the nearest choice.
- **Try it:** `../.venv/bin/python -m ml show-actions --seed 3`
  (add `--seconds 20` for a mid-game position).
- **Tests:** `../.venv/bin/python -m unittest ml.tests.test_action_encoder`
- **Why:** the raw action (any edge, any integer) is too large to learn
  directly. Masking means the network never wastes effort on illegal moves.
- **Concept:** factored action spaces and action masking.
- [x] Legal choices, decode, and teacher-move encode
- [x] Test: every offered choice is accepted by `Game._validate_action`, and
      every route the game accepts is offered
- [x] Test: "just enough" really captures in the real game, and one unit
      fewer does not
- [x] Coverage check against the scripted AIs (see "Measurements"); it moved
      the default from quarters to eighths
- [x] Demo script (`show_actions.py`)
- [x] Your review of the amount options (2026-09-30)
- **Known simplifications:**
  - "Just enough" covers capturing, not defending. Reinforcing a base before
    an attack lands has to be sized with the fraction options.
  - It counts only fleets already in flight; the defender can still respond.
  - One send per decision, as the game allows. Moving a large army through
    several bases takes several decisions.
- **Done when:** random legal choices play thousands of ticks with zero
  rejections (met) and you are happy with the amount options.

### Stage 4 — Network and checkpoints (`models/`) ✅
- **What:** a small transformer over bases (120,120 weights) with three
  outputs: which route (or wait), how much, and who is winning.
- **Swappable by design:** every architecture takes a `ModelInput` and returns
  a `ModelOutput` (`models/interface.py`). An architecture only produces
  scores; removing illegal choices happens in the shared base class, and
  sampling, probabilities, and entropy live in `models/policy.py`. Adding an
  architecture is one new file plus one line in `models/__init__.py`; it is
  then chosen by name with `build_model("name")`.
- **Try it:** `../.venv/bin/python -m ml show-model --seed 3` prints
  the weight count per part and what the untrained network would do.
- **Tests:** `../.venv/bin/python -m unittest ml.tests.test_models`
- **Why:** attention lets every base see every other base in one step, which
  matters for moving armies across a large map. One network with shared
  understanding feeds both "what to do" (policy) and "how am I doing" (value).
- **Concept:** policy and value heads, attention, and what a checkpoint is.
- [x] Model interface, policy helpers, and registry
- [x] First architecture (`graph_transformer`) with a config object
- [x] Tests: reordering bases reorders the outputs the same way; padding does
      not change the answer; illegal choices have zero probability; all
      probabilities sum to one; every weight receives a learning signal; a
      second toy architecture works through the same interface
- [x] Demo script (`show_model.py`)
- [x] Decision on the legal-choice table as an input: only the
      `can_send_just_enough` flag per route is used (see "Decisions")
- [x] Save, load, and `init_model("random" | path)` (`checkpoint.py`). A
      checkpoint holds the architecture name and settings, the weights, a
      history of the steps that produced it, optional training state, and the
      encoders' feature lists; loading is refused if those lists have changed.
- [x] Command line: `python -m ml checkpoint new PATH --seed 0` and
      `python -m ml checkpoint show PATH`; `show_model` accepts `--init PATH`
- [x] Your review of the model and checkpoints (2026-09-30)
- **Done when:** a forward pass works on every map size and round-trips
  through a checkpoint.

### Stage 5 — Agent and evaluation (`play/agent.py`, `evaluation/evaluate.py`, `ais/neural_ai.py`) ✅
- **What:** wrap the network as `choose_action(state)` and measure it against
  a fixed pool of opponents on maps it has not trained on.
- **Why:** we need the ruler before we start training, otherwise "it got
  better" is a feeling. An untrained network gives us the zero point.
- **Concept:** baselines and win rate with uncertainty. (Elo moves to Stage 9:
  it needs several networks to rate against each other, and is undefined for
  a player that almost never wins.)
- **Watch it play:** `../.venv/bin/python main.py`, then pick "Neural AI" for
  a seat; or `main.py --map classic --red-ai neural_ai`. It plays
  `ml/runs/latest.pt`, or the file named in `NEURAL_AI_CHECKPOINT`, or an
  untrained network if neither exists. The file is re-read at the start of
  every match.
- **Measure it:** `../.venv/bin/python -m ml evaluate --candidate untrained`
  (or a checkpoint path, or an AI name; `--quick` for a 10-second check).
- **Tests:** `../.venv/bin/python -m unittest ml.tests.test_agent ml.tests.test_evaluate`
- [x] Agent wrapper (`agent.py`) and `ais/neural_ai.py`, selectable from the
      game's home screen
- [x] Evaluator: 8 bundled 1v1 maps + 30 generated maps from reserved seeds,
      both seats, live progress; out of the game counts as a loss
- [x] Baselines recorded (see "Baselines")
- [x] Your review (2026-09-30)
- **Changed outside `ml/`:** new files `ais/neural_ai.py` and (Stage 11)
  `ais/search_ai.py`; `tests/test_game.py` lists them among the expected AIs.
- **Done when:** you can watch an untrained network play in the UI, and its
  numbers are in the table below.

### Stage 6 — Rollouts (`play/rollout.py`) ✅
- **What:** play many games at once and record, for every decision, what was
  seen, what was done, and what happened next.
- **Why:** both trainers need this, so it is built once. Batching matters
  because the network is ~10x faster per decision when it answers hundreds of
  positions together.
- **Concept:** trajectories, batched environments, the speed budget.
- **How it is used:** a trainer lists matchups (a map and who sits in each
  seat) and gets back one record per game. Each seat is an AI file or a
  model, and each seat marked `record` yields a trajectory: every decision's
  encoded position, legal choices, and the choice made. `RolloutPool` does
  the same on several cores and stays alive between training rounds.
- **Try it:** `../.venv/bin/python -m ml rollout --red untrained --blue strategic_ai --games 64 --workers 6`
- **Tests:** `../.venv/bin/python -m unittest ml.tests.test_rollout`
- [x] Batched game runner with pluggable players (model or scripted), in one
      process (`play_games`) or several (`RolloutPool`)
- [x] Your review (2026-09-30)
- [x] Trajectories for both trainers: a model's choices with its probability
      and value at the time, or a teacher's moves mapped to the nearest choice
- [x] Every game keeps its full move list; replaying it gives the same
      winner and length, and every stored decision matches the position it
      was made in
- [x] Throughput measured and the two biggest costs cut (see "Measurements")
- [x] `decision_interval` setting: model players decide every Nth tick
- **Changes to earlier stages, made for speed (results unchanged, tests
  still pass):**
  - `forecast.py` now works out each base's future once per position
    (`BaseTimeline`) and both encoders and both seats share it.
  - Both encoders fill their arrays in one step instead of row by row.
  - `graph_transformer` scores only the routes that can be used, not every
    pair of bases.
- **Done when:** recorded games replay to the identical outcome (met).

### Stage 7 — Imitation trainer (`training/imitation.py`) ✅
- **What:** record the teacher's games, then train the network to predict the
  teacher's move and the eventual winner.
- **Why:** it is ordinary supervised learning, so it is fast and easy to
  debug. If the network cannot even copy a teacher, the encoder or action
  space is broken, and it is far cheaper to find that out here than in RL.
- **Concept:** behaviour cloning, loss curves, and the ceiling: a copy is at
  best as good as its teacher.
- **Run it:** `../.venv/bin/python -m ml train imitation --run NAME`
  (`--help` lists every setting with its default; `--monitor` opens the
  charts alongside). Settings live in `ImitationSettings` at the top of
  `imitation.py`.
- **Watch it:** `../.venv/bin/python -m ml monitor --run NAME`, at any time.
- **Tests:** `../.venv/bin/python -m unittest ml.tests.test_dataset ml.tests.test_runlog ml.tests.test_imitation ml.tests.test_monitor`
- [x] `dataset.py`: recorded decisions as shuffled batches, grouped by map
      size; shared with the self-play trainer
- [x] Teacher games: 600 by default, the teacher taking each seat equally,
      against itself and the two simpler AIs; at most 300 decisions kept per
      seat and game; one game in ten held out for checking
- [x] Training loop: move error plus value error, step size shrinking over
      the run, measured every epoch on the held-out games
- [x] Matches against the scripted AIs after every epoch, and a full
      evaluation of the best model at the end
- [x] Live feedback: one terminal line per epoch, `metrics.jsonl`, and the
      monitor window (`monitor.py`)
- [x] A checkpoint every epoch, plus `latest.pt` and `best.pt`
- [x] First real run (`imitation_v1`, see "Baselines")
- [x] Your review (2026-09-30)
- **Done when:** the clone is roughly even with `strategic_ai` on unseen maps
  (met: 33 wins, 37 losses, 6 draws in 76 games).

### Stage 8 — Self-play trainer (`training/selfplay.py`) ✅
- **What:** the network plays itself; moves from won games become more
  likely, moves from lost games less likely (PPO).
- **Why:** this is the part that can exceed any teacher, because the target is
  winning rather than copying.
- **Concept:** rewards and returns, credit assignment over long games, the
  policy gradient, why PPO limits the size of each update, reward shaping.
- **Run it:** `../.venv/bin/python -m ml train selfplay --run NAME`, adding
  `--init PATH` to continue from a checkpoint such as an imitation model.
  Settings live in `SelfPlaySettings` at the top of `selfplay.py`.
- **Tests:** `../.venv/bin/python -m unittest ml.tests.test_selfplay`
- **One round:** play 96 games with the current network in both seats; judge
  every move; go over a sample of the decisions up to four times; save; every
  fifth round, play the scripted AIs and the model the run started with.
- [x] Result: +1 win, -1 loss, 0 draw or timeout. The value output learns the
      result, discounted by how far away the end was.
- [x] Judging moves (`judge_moves`): each move is credited with how much
      better things looked afterwards than expected, nearer surprises counting
      more (generalised advantage estimation, measured in game ticks because
      decisions are not evenly spaced)
- [x] The material rule of thumb: until the value output can judge
      positions, a larger share of units and production counts as better. It
      is added to the expectation, not to the result, so it cannot change
      what winning means, and it fades to zero over the first 50 rounds.
- [x] PPO update: clipped probability change, entropy bonus, and a stop as
      soon as the policy has moved far enough (checked after every batch)
- [x] Value-only first rounds, with the moves actively held in place
- [x] Step size that adjusts itself between rounds to how far the policy
      actually moved
- [x] Optional scripted opponents in training (`--opponents self strategic_ai`)
- [x] Feedback: a terminal line and chart data every round; evaluation
      every fifth round on the rollout workers, so no second set of
      processes is started
- [x] Round 0: both trainers evaluate the starting model before any
      training, so every score chart begins at the true starting point
- [x] Value error is charted next to what "always saying 0" would score on
      the same games, since the raw number is hard to read alone
- [x] Real run from random weights (`selfplay_scratch_v1`, see "Baselines")
- [x] Real run from the imitation model (`selfplay_from_imitation_v1`, see "Baselines")
- [x] Your review (2026-09-30)
- **Done when:** win rate against a frozen copy of itself rises above 50% and
  keeps rising (met from random weights: 50% before training, 88% at round
  10, 100% at round 30).

### Stage 9 — League (`training/league.py`) — built, in review
- **What:** keep past versions as opponents alongside the scripted AIs, and
  only promote a new version when it beats the current best.
- **Why:** pure self-play can go in circles (A beats B beats C beats A).
  Old versions and scripted AIs keep it honest, and promotion is what makes
  "improves over time" a guarantee instead of a hope.
- **Concept:** non-stationarity, forgetting, gating, rating scales.
- **Run it:** `../.venv/bin/python -m ml train league --run NAME --init PATH`.
  Settings live in `LeagueSettings` at the top of `league.py`; everything
  about playing, judging, and learning is shared with self-play.
- **Tests:** `../.venv/bin/python -m unittest ml.tests.test_league`
- **One round:** half the games against itself, 30% against past champions,
  20% against the scripted AIs; among champions and among scripted AIs, the
  ones it has recently done worst against are met most often.
- **Every fifth round:** a 128-game match against the reigning champion on
  fresh maps, from both seats. At 55% or more it becomes the new champion.
  Only champions are saved as `best.pt`.
- [x] The training loop is shared: self-play and the league differ only in
      who supplies the opponents (`SelfPlayOpponents` and `League`)
- [x] Champion pool (newest eight stay as opponents), opponent sampling that
      favours the ones it struggles against
- [x] Promotion match and gate
- [x] Ratings (Elo) fitted from every match between league members at once:
      scripted AIs against each other, each champion against the scripted AIs
      and against its predecessor. The scripted AIs average 1000 by definition.
- [x] `league.json` in the run folder: champions, ratings, every promotion
      match, every result; and a league table printed at the end
- [x] Monitor: up to nine charts, three to a row
- [x] First real run (`league_from_imitation_v1`): promotions at rounds 15
      and 25, three failed matches
- [x] `ml.evaluation.evaluate --skip-generated N`: evaluate on generated maps that no
      trainer used to choose a best model
- [x] Resume (2026-10-01): `--stop-after N` and `--resume` work for the league
      too. Every round's checkpoint carries the league's memory (champions as
      names and rounds, all results, recent scores, promotion matches); the
      champions' weights are reloaded from their `round_XXXX.pt` files, so
      those must be kept while the run goes on. Ratings are refitted on load.
- [ ] Your review
- **Done when:** the rating of the champions only goes up (met: 1233, 1264,
  1324 for champions 0, 1, 2).

### Stage 10 — The experiment: random start vs. imitation start — paused, to be folded into the long run
- **What:** run Route A and Route B with the same budget, maps, and opponent
  pool, several seeds each, and compare the curves.
- **Why:** this is the question you asked. The interesting outcomes are not
  only "which is stronger" but also "which is faster" and "do they end up
  playing differently".
- **Design** (`ml/scripts/done/stage10.sh`):
  - Both routes: 150 rounds of plain self-play, default settings, three
    seeds each. Seed 0 of the random start is the first 150 rounds of the
    overnight run; the other five runs are new.
  - The imitation start begins from the last epoch of an imitation run
    (600 teacher games, 20 epochs), one imitation run per seed. Its time
    counts towards the imitation route.
  - Rounds 0, 25, 50, 100, and 150 of every run are rated on one scale
    (`ml/evaluation/compare.py`): each plays the three scripted AIs and four overnight
    checkpoints (rounds 100, 200, 300, 500), 100 games each, on fresh maps;
    the final checkpoints of the two routes also play each other.
  - Expected time: about 7 hours.
- **Paused on 2026-10-01 at 10:44**, at your request, during the second
  imitation run. `ml/scripts/done/stage10.sh` can still carry on where it stopped.
- **Proposed instead:** answer the question on the network we will keep. The
  long run is done twice, once from random weights and once from imitation,
  with the deeper network (see "Next steps"). That costs fewer seeds, so
  differences under about 50 rating points will not be visible.
- [x] Comparison tool (`ml/evaluation/compare.py`), with tests
- [x] Both routes on the deep network, one seed each (2026-10-02), below
- [ ] More seeds (one each so far)
- [ ] Curves: strength vs. games played, and vs. wall-clock time
- [ ] Short write-up of what we saw, including anything that surprised us

**Result on the deep network (2026-10-02)**, `ml/runs/compare_stage10_deep/report.txt`.
Both routes: 64×6, 500 rounds of plain self-play, default settings. The
imitation route first copied `strategic_ai` for 20 epochs (11 minutes;
`deep_imitation`, 74% of its sends copied), which counts towards its time.

| Round | Imitation start: rating (hours) | Random start: rating (hours) | vs `strategic_ai`, imitation / random |
|---|---|---|---|
| 0 | 969 (0.2) | 400 (0.0) | 41% / 3% |
| 50 | 1091 (0.7) | 888 (0.6) | 63% / 38% |
| 100 | 1150 (1.3) | 1067 (1.1) | 77% / 64% |
| 200 | 1163 (2.5) | 1162 (2.2) | 74% / 72% |
| 300 | 1261 (3.6) | 1207 (3.3) | 82% / 75% |
| 400 | 1323 (4.6) | 1254 (4.4) | 86% / 77% |
| 500 | **1357** (5.8) | 1263 (5.6) | 84% / 80% |

- **At round 500, the imitation start is ahead:** about +95 points, and
  65.5 of 100 games head to head (3.1 standard errors). That is the state
  at this budget, not a verdict on which route ends higher.
- **The rates of progress do not separate the routes** (your reading,
  2026-10-02): rounds 100–300 gained +111 (imitation) and +140 (random),
  rounds 300–500 +96 and +56. Single checkpoints are good to about ± 30
  points, so differences in gains of this size are within the noise. The
  random start may simply be somewhat behind on the same path. Whether it
  catches up with more training is still open, as in the prediction.
- **Cost of the head start:** 11 minutes of imitation and 600 games of
  `strategic_ai`. Its rounds were slower (5.8 h against 5.6 h in all).
- **Caveat:** one seed per route; seeds have differed by up to about 50
  points. Longer runs (and more seeds) are what would settle it.
- `deep_from_imitation` round 500 (1357) is the strongest network so far.

**Move scoring on the deep model** (`value:deep_scratch/round_0500.pt?candidates=8`,
2026-10-02, log `ml/runs/logs/after_deep_log.txt`): 20 fresh maps from both
seats, 40 games per opponent, the original Atlas's opponent network pinned
to the same deep model.

| Against | Score | Wins / draws / losses |
|---|---|---|
| `careful_ai` | 97.5% | 39 / 0 / 1 |
| `aggressive_ai` | 96.2% | 38 / 1 / 1 |
| `strategic_ai` | 77.5% | 30 / 2 / 8 |
| first overnight model (64×3, round 500) | 75.0% | 29 / 2 / 9 |
| **the original Atlas** | **61.3%** | **23 / 3 / 14** |
| Atlas Clean | 61.3% | 24 / 1 / 15 |
| the same deep model, plain | 57.5% | 22 / 2 / 16 |

- **For the first time, one of ours beat the original Atlas**, which had
  beaten the first overnight model 12–4. At 40 games this is about 1.5
  standard errors from even: likely, not proven.
- Move scoring added less on the deep network than on 64×3 (57.5% against
  64.4% over the plain network), but 40 games are too few to tell those
  apart.
- Not yet tested: move scoring on the stronger imitation-start model.
### Stage 11 — Search and distillation (`play/search.py`, `training/distill.py`) — built, in review
- **What:** before moving, try the network's top few candidate moves in the
  exact simulator, look a few seconds ahead, and pick the best; then train the
  network to choose that move directly.
- **Why:** this is the affordable core of AlphaZero. Atlas, which beats our
  network, looks ahead in the same way (with our network inside its
  look-ahead; see "Atlas as a yardstick").
- **How it plays** (`SearchAgent`): every third tick, unless the network is
  at least 90% sure it should wait, it takes waiting plus the network's four
  favourite sends, plays each out for 6 game seconds in a copy of the game
  (both sides played by the network, deciding every second tick), asks the
  value output who is winning at the end, and plays the best. Other ticks
  are played on instinct. Settings in `SearchSettings`.
- **Use it:** anywhere a player is named, write `search:PATH` (evaluator,
  rollouts, trainers); in the game, pick "Search AI" (`ais/search_ai.py`,
  checkpoint from `SEARCH_AI_CHECKPOINT`, `NEURAL_AI_CHECKPOINT`, or
  `ml/runs/latest.pt`).
- **Distillation, one step by hand:** the imitation trainer with the
  searching network as the teacher, starting from the same network. As a
  teacher, the search agent marks its instinct moves so that only searched
  decisions are recorded.

      ../.venv/bin/python -m ml train imitation --run distilled --init PATH \
          --teacher search:PATH --opponents search:PATH --games 60 --epochs 5 \
          --learning-rate 0.0001 --final-learning-rate 0.00001

- **Distillation, the full loop** (`training/distill.py`, `ml train distill`):
  each iteration (1) lets the newest network, searching, play itself
  (`games_per_iteration`, default 100), recording only searched moves;
  (2) trains on those plus the previous iteration's decisions
  (`remembered_iterations`, default 2) for a few epochs, continuing from the
  newest network with a fresh small step size; (3) checks the result on
  fresh maps against the previous iteration, the starting model, and
  `strategic_ai` (`match_games`, default 64). With `keep_threshold` above 0,
  an iteration that does not beat its predecessor by that margin is thrown
  away. Every iteration is saved as a round, so the monitor, `--resume`,
  `--stop-after`, and `ml compare` work as for self-play. The search
  settings travel in the player name (`search:PATH?candidates=6&horizon_seconds=4.0`).

      ../.venv/bin/python -m ml train distill --run distilled \
          --init ml/runs/overnight_scratch/latest.pt --iterations 10

  Expected cost: about 25–30 minutes per iteration with 100 games on six
  cores (the 60-game step above took 16 minutes). Not run for real yet, only in the tests and a short check.

- **Tests:** `../.venv/bin/python -m unittest ml.tests.test_search`
- [x] A copy of the game built from a view plays on exactly like the real game
- [x] Search agent, `search:` players, "Search AI" in the game
- [x] Distillation through the imitation trainer
- [x] The full loop as its own trainer, with tests
- [ ] A real run of several iterations
- [ ] Your review

**Short checks (2026-10-01)**, overnight model (round 500), 16 fresh maps
from both seats, 32 games per row, so each score is good to about ± 16:

| Player | vs plain overnight model | vs `strategic_ai` |
|---|---|---|
| plain overnight model | (itself) | 62.5% (20 wins, 12 losses) |
| the same model with search | 68.8% (21 wins, 9 losses, 2 draws) | 75.0% (24 wins, 8 losses) |
| after one distillation step | 60.9% (19 wins, 12 losses, 1 draw) | 54.7% (17 wins, 14 losses, 1 draw) |

- **Search helps:** it beat the same network without search 21 to 9, about
  2 standard errors from even. Cost: about 30 ms per tick (15 to 30 seconds
  per game), against about 1 ms for the plain network.
- **One distillation step is inconclusive.** 60 games of the searching
  network against itself (16 minutes, 24,805 recorded decisions), 5 epochs
  at a small step size. The network's most likely move matched the search's
  choice 11% of the time before and 22% after. It then beat the original
  61% but did worse against `strategic_ai`; both are within the noise.
- Distillation needs far more searched games than this to show an effect,
  and each searched game costs about a minute of one core. That is the
  price of this route; a long run of it is an overnight job.

## Housekeeping (2026-10-01)

- **Code regrouped** into `play/`, `training/`, and `evaluation/`, with
  `checkpoint.py` moved into `models/` (see "Layout"). All commands now run
  as `python -m ml <command>`; the old `python -m ml.train` style no longer
  works. The three `ais/_neural_*reference.py` files of Atlas import
  `ml.play.agent` now (a one-line change each).
- **Runs removed** (their numbers stay in this file): `selfplay_scratch_v1`,
  `selfplay_from_imitation_v1`, `league_from_imitation_v1`, `distill_check`,
  the unfinished `stage10_imitation_seed2` (Stage 10 redoes it), the
  individual runs inside both sweeps (their reports and results remain), and
  all tuning scripts and logs.
- **Runs thinned out:** `overnight_scratch` keeps every 25th round (0, 25,
  …, 500), its latest and best, and its full metrics; `imitation_v1` and
  `stage10_imitation_seed1` keep their final epoch (and round 0 where there
  is one). Everything Stage 10 needs is still there. `ml/runs/` went from
  532 MB to 19 MB.

## Baselines

Score counts a win as 1 and a draw as 1/2. 228 games each: 38 maps, both
seats, three opponents, 180 s limit. The ± is the range the score would
likely land in if replayed.

| Candidate | vs aggressive | vs careful | vs strategic | Overall score | Wins / draws / losses |
|---|---|---|---|---|---|
| `strategic_ai` | 85.5% | 94.7% | 50.0% | 76.8% ± 5.5% | 157 / 36 / 35 |
| `careful_ai` | 77.0% | 50.0% | 5.3% | 44.1% ± 6.4% | 69 / 63 / 96 |
| `aggressive_ai` | 50.0% | 23.0% | 14.5% | 29.2% ± 5.9% | 46 / 41 / 141 |
| imitation, then 30 rounds of self-play (`selfplay_from_imitation_v1`) | 84.9% | 95.4% | 61.8% | 80.7% ± 5.1% | 178 / 12 / 38 |
| imitation, then a 30-round league (`league_from_imitation_v1`, champion 2) | 82.2% | 93.4% | 45.4% | 73.7% ± 5.7% | 165 / 6 / 57 |
| imitation of `strategic_ai` (`imitation_v1`, best epoch) | 79.6% | 88.2% | 47.4% | 71.7% ± 5.8% | 158 / 11 / 59 |
| self-play from random weights, 30 rounds (`selfplay_scratch_v1`) | 75.7% | 65.8% | 27.6% | 56.4% ± 6.4% | 122 / 13 / 93 |
| untrained network (2026-09-30) | 9.2% | 16.4% | 3.9% | 9.9% ± 4.0% | 8 / 29 / 191 |
| untrained network, greedy | 23.7% | 45.4% | 0.0% | 23.0% ± 5.5% | 0 / 105 / 123 |

### On fresh maps

The table above shares 8 of its 30 generated maps with the quick checks that
choose each run's "best" model, which flatters the chosen model. This table
uses 60 generated maps that took no part in any choice (`--no-bundled-maps
--skip-generated 30 --generated-maps 60`): 360 games each, about ± 4.5.

| Candidate | vs aggressive | vs careful | vs strategic | Overall | Wins / draws / losses |
|---|---|---|---|---|---|
| overnight self-play from random weights, round 500 (`overnight_scratch`) | 95.8% | 91.7% | 72.9% | 86.8% | 308 / 9 / 43 |
| `strategic_ai` | 85.4% | 89.6% | 50.0% | 75.0% | 237 / 66 / 57 |
| self-play from imitation, round 30 | 88.8% | 88.3% | 57.9% | 78.3% | 271 / 22 / 67 |
| self-play from imitation, "best" (round 20) | 89.2% | 90.0% | 52.9% | 77.4% | 266 / 25 / 69 |
| league from imitation, champion 2 (round 25) | 87.5% | 89.2% | 53.8% | 76.8% | 265 / 23 / 72 |
| league from imitation, round 30 | 85.8% | 88.3% | 50.8% | 75.0% | 258 / 24 / 78 |
| imitation (`imitation_v1`) | 85.4% | 84.6% | 41.2% | 70.4% | 239 / 29 / 92 |
| self-play from random weights (`selfplay_scratch_v1`) | 76.2% | 65.8% | 24.2% | 55.4% | 183 / 33 / 144 |

Reading: 30 rounds of self-play or league took the imitation model from
about 70% to about 75–78%, which is level with its teacher. Against
`strategic_ai` itself it went from about 41% to about 51–58%. Plain self-play
and the league cannot be told apart after 30 rounds. The 61.8% against
`strategic_ai` in the first table was the top of the noise: the same model
scores 52.9% here.

Caution: `selfplay_scratch_v1` was trained before the step size adjusted
itself (fixed at 3e-4, a policy change of about 0.009 per round, so
cautious), so it is not comparable like for like with later runs. Each "best"
model was picked by a quick check on 8 of the 30 generated evaluation maps,
which flatters it slightly on those maps. The run folders of these rows were
deleted in the housekeeping; the numbers stay here.

Caution: score rewards stalling. The greedy untrained network never wins a
game, yet scores 23% because `careful_ai` and `aggressive_ai` often fail to
finish off a passive opponent. Always read wins alongside score.

## Tuning

**Question:** which settings to use for the long runs.

**Method** (`ml/evaluation/sweep.py`):
- Train the same starting model once per variation of the settings, each
  variation with two seeds, 20 rounds each.
- Let every result play the starting model, the three scripted AIs, and the
  results of the other variations, on generated maps that no trainer or
  earlier evaluation has used (from map 100 on). All matches together give
  each run one rating.
- The gap between two seeds of the same variation is the noise. A variation
  only counts as better if it beats that.
- One setting is changed at a time. Settings without a specific reason to
  doubt them are left alone.

**Plan** (the sweep script and its log were removed in the housekeeping; the reports remain):

| Sweep | Starts from | Variations | Why |
|---|---|---|---|
| `from_imitation` | the imitation model | `entropy_weight` 0.01 / 0.003 / 0; and larger steps per round (`target_policy_change` 0.03 with `max_policy_change` 0.06) | Entropy rose from 0.46 to over 1.0 in both earlier runs from the imitation model; and the step size settled low |
| `from_scratch` | random weights | `material_hint` 0.5 / 0 | Whether the rule of thumb helps, and the first test of the self-adjusting step size from random weights |

**A problem found on the way, and fixed.** The first attempt slowed down
run by run: 11 minutes for the first run, 20 for the second, 64 for the
third, and the fourth was taking about 6 minutes per round. The sweep's
process had grown to 24 GB and the machine was swapping 34 GB.

- Cause, measured: on the Apple GPU, every new tensor shape used in
  learning costs memory that is never given back (about 8–12 MB each). Two
  shapes changed constantly: the size of each group's last batch, which is
  different every round, and the number of usable routes in each batch.
  Running the model forward with changing shapes cost nothing; only
  learning did.
- Fix: self-play batches are now always full (a short last batch is topped
  up with decisions drawn again from its group), and the model pads its list
  of usable routes to one of a few fixed lengths. Over 24 simulated rounds
  memory now stays at 3.74–3.76 GB, where before it grew by about 70–240 MB
  per round.
- This also explains why the earlier 30-round runs slowed towards the end
  (the self-play run from imitation went from about 40 s per round to about
  3 minutes). Their results are unaffected; only their speed was.
- After the fix, memory still grew by about 70 MB per round in the real
  sweep (new combinations of map size and route count keep appearing). So
  the sweep now also trains every run in a fresh process, which returns all
  memory between runs. In a single long run it does not level off (about 40
  MB per round), so long runs restart every 50 rounds (see "Overnight run").
- The sweep was restarted. The three runs finished before the fix are kept:
  the fix changes the order of work, not what is learned, apart from a few
  decisions per round being seen twice.

**Results, sweep `from_imitation`** (20 rounds of self-play from the imitation
model, two seeds per variation, 240 games per match on fresh maps; full
table in `ml/runs/sweep_from_imitation/report.txt`):

| Variation | Rating, mean | Lowest–highest of its two runs | vs starting model | vs `strategic_ai` | Entropy at the end |
|---|---|---|---|---|---|
| larger steps (`target_policy_change` 0.03, `max_policy_change` 0.06) | 1253 | 1250–1256 | 55.5% | 53.5% | 1.06 |
| `entropy_weight` 0.01 (default) | 1248 | 1237–1259 | 56.1% | 51.0% | 0.79 |
| `entropy_weight` 0.003 | 1236 | 1218–1255 | 54.7% | 45.5% | 0.63 |
| `entropy_weight` 0 | 1236 | 1209–1262 | 57.3% | 51.0% | 0.71 |

On this scale `strategic_ai` is 1241 and the starting model 1202.

- **Noise floor:** two runs with identical settings ended up to 53 points
  apart. The largest gap between variation averages is 17 points. Direct
  matches between variations all landed between 46% and 55% over 240 games,
  which is what chance alone produces. **No variation is measurably better.**
- **Every variation improved** on the starting model, by about 35–50 points
  (55–57% in direct matches), and ends level with `strategic_ai`.
- **Entropy rises even without the entropy bonus** (0.46 to 0.71 at weight 0).
  The earlier worry, that the bonus pushes a confident copy to become
  undecided, does not hold: the rise comes from self-play itself.
- **Consequence for the defaults:** keep them. Nothing here justifies a change.
- **Consequence for Stage 10:** differences smaller than about 50 points need
  more than two seeds per variation to be seen at all.

**Results, sweep `from_scratch`, first two seeds** (20 rounds of self-play from
random weights; `ml/runs/sweep_from_scratch/report_two_seeds.txt`):

| Variation | Rating, mean | Lowest–highest | vs `aggressive_ai` | vs `careful_ai` | vs `strategic_ai` |
|---|---|---|---|---|---|
| `material_hint` 0 (no rule of thumb) | 915 | 898–931 | 59.0% | 51.0% | 12.1% |
| `material_hint` 0.5 (default) | 881 | 862–901 | 51.0% | 48.1% | 12.9% |

On this scale `careful_ai` is 901 and `aggressive_ai` 863.

- In direct matches the rule of thumb scored 42.3% over 240 games against
  no rule of thumb, about 2.4 standard errors below even. The ratings point
  the same way, but the gap (34) is similar to the spread between seeds
  (33–39). **Suggestive, not settled:** the rule of thumb does not help and
  may hurt.
- **The self-adjusting step size climbs to its ceiling (0.001) by round 9
  from random weights and stays there**, while the policy still moves less
  than the target per round. The ceiling is what limits learning speed here.
- A third seed and a run with a higher ceiling (0.003) were added; see below.

**Results, sweep `from_scratch`, three seeds** (`ml/runs/sweep_from_scratch/report.txt`):

| Variation | Rating, mean | Lowest–highest of three runs | vs `aggressive_ai` | vs `careful_ai` | vs `strategic_ai` |
|---|---|---|---|---|---|
| `material_hint` 0 | 897 | 856–938 | 53.8% | 49.8% | 13.1% |
| `material_hint` 0.5 (default) | 892 | 860–918 | 53.3% | 49.6% | 11.5% |
| `material_hint` 0 with `max_learning_rate` 0.003 | 806 | 766–830 | 39.5% | 39.2% | 8.3% |

- **The rule of thumb makes no difference.** With the third seed the gap
  vanished (5 points) and direct matches came out at 48.9% over 540 games.
  The two-seed result was chance: a useful lesson in why seeds matter.
- **A higher step-size ceiling is clearly worse.** Both other variations beat
  it in direct matches, 61% and 63% over 540 games (more than 5 standard
  errors). The ceiling being reached is not a sign that it is too low.

**Conclusion of the tuning:** keep the current defaults. Nothing tested beat
them, and one change (a higher step-size ceiling) was clearly worse. The
defaults are used for the overnight run.

## Overnight run (started 2026-10-01 00:38)

- **What:** self-play only, from random weights, 500 rounds, default
  settings, evaluation every 10 rounds. Run `overnight_scratch`.
- **How:** `ml/scripts/done/overnight_scratch.sh` trains at most 50 rounds per
  process and carries on in a fresh one with `--resume`, because a single
  process still grows by about 40 MB per round.
- **Watch:** `../.venv/bin/python -m ml monitor --run overnight_scratch`
- **Stop:** `pkill -f overnight_scratch.sh; pkill -f "ml train selfplay --run overnight_scratch"`
- **Carry on after stopping:** run `ml/scripts/done/overnight_scratch.sh` again.
- **Result:** finished at 04:26 after 3 h 48 min, without problems. Five
  restarts between processes, memory steady at 5.3–5.5 GB.

| Round | Quick check: vs `aggressive_ai` / `careful_ai` / `strategic_ai` (16 games each) |
|---|---|
| 0 | 9% / 19% / 0% |
| 60 | 88% / 88% / 12% |
| 160 | 100% / 100% / 50% |
| 270 | 100% / 94% / 94% |
| 400 | 100% / 100% / 56% |
| 500 | 100% / 88% / 94% |

On fresh maps (360 games each, the same maps as in "On fresh maps"):

| Model | vs aggressive | vs careful | vs strategic | Overall | Wins / draws / losses |
|---|---|---|---|---|---|
| round 500 | 95.8% | 91.7% | 72.9% | 86.8% | 308 / 9 / 43 |
| "best" (round 270, picked by quick check) | 96.2% | 90.0% | 67.1% | 84.4% | 297 / 14 / 49 |

- Against `strategic_ai` the final model won 85 games and lost 30 of 120.
  It is now clearly stronger than the strongest scripted AI, and stronger
  than every earlier model, including those that started from imitation.
- The last round beat the "best" round. The quick check that picks "best"
  is too noisy for long runs; the final model is the one to use.
- `ml/runs/latest.pt`, which the game's Neural AI plays, is round 500.

**Is it still improving?** Checkpoints from rounds 50 to 500 and
`strategic_ai` played a round-robin on fresh maps (200 games per pair, 4,200
games, rated together with `strategic_ai` fixed at 1000):

| Round | 50 | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|---|
| Rating | 761 | 921 | 1067 | 1159 | 1172 | 1195 |
| vs `strategic_ai` | 28% | 46% | 63% | 67% | 65% | 70% |

| Later against earlier | Score |
|---|---|
| round 200 vs round 100 | 77% |
| round 300 vs round 200 | 67% |
| round 400 vs round 300 | 53% |
| round 500 vs round 400 | 51% |
| round 500 vs round 300 | 56% |

- Nearly all the progress came in the first 300 rounds (about 2.3 hours):
  +400 rating points. The next 200 rounds added about 35, and round 500 is
  level with round 400 head to head.
- With these settings the run has reached a plateau. More rounds of the same
  would most likely add little: 10,000 more rounds is about 3 days of
  training for perhaps a few dozen points, if any.
- Possible reasons, untested: the network's size (120,120 weights), playing
  only against its current self, or the step size, which has shrunk to about
  7e-5 as the policy became confident.

**Is the small step size the problem? (reviewed 2026-10-01): no, not by
itself.** The learning rate is not set by hand. It is a dial that the
trainer turns each round so that the policy moves by a fixed amount
(`target_policy_change` 0.015, kept between 0.010 and 0.0225).

| Rounds | Learning rate (median) | Policy change per round (median) | Share of moves at the clip limit | Entropy |
|---|---|---|---|---|
| 1–25 | 1.0e-3 (at the ceiling) | 0.011 | 12% | 4.35 |
| 51–75 | 3.0e-4 | 0.017 | 19% | 3.82 |
| 176–200 | 8.8e-5 | 0.014 | 13% | 3.38 |
| 326–350 | 8.8e-5 | 0.017 | 14% | 3.18 |
| 476–500 | 8.8e-5 | 0.017 | 14% | 3.17 |

- From round 25 on, the policy moved by the target amount in every window,
  whatever the learning rate was. Round 450 changed the policy as much as
  round 60 did, so a learning rate of 9e-5 did not mean less learning per
  round. The rate fell because the trained network reacts more strongly to
  the same change in its weights; it never reached the floor of 1e-5.
- So the learning rate does not limit progress per round. The policy-change
  target does. Raising the learning rate by hand would just make the
  controller lower it again, or would push past the target. Doubling the
  target did nothing measurable in the 20-round tuning sweep from imitation,
  but that was early in training, not at a plateau.
- The plateau is more likely to come from somewhere else: playing only
  against itself, noisy advantages from 96 games per round, or the
  network's size. The policy is still wide (entropy 3.2) after 500 rounds.
- Side check, 80 games each on fresh maps: round 500 scored 69% against
  `strategic_ai` with sampled moves and 78% with greedy moves (the
  difference is within the noise), and 48% against round 400.

## Atlas as a yardstick (from 2026-10-01)

`ais/atlas_ai.py`, the search-based AI from the research notes, is installed.

| Atlas against | Average per decision | Slowest decision | One 18-base game |
|---|---|---|---|
| `strategic_ai` | 13 ms | 3.6 s | 11 s |
| our network | 220 ms | 10.7 s | about 3 minutes |

Against a network it simulates nine candidate plans three times each, 60
seconds ahead, with our network inside the simulation predicting our moves.

- **Use:** occasional evaluations only. Too slow as a training opponent.
  Possible as an imitation teacher (its games against scripted AIs are fast).
- **It predicts us from a checkpoint file:** `ATLAS_OPPONENT_CHECKPOINT`, else
  `NEURAL_AI_CHECKPOINT`, else `ml/runs/latest.pt`, which training overwrites
  every round. Evaluations pin it explicitly; pinning it to the candidate
  itself is the hardest test (Atlas knows the network, not its dice).
- **It has its own copy of our encoders** (`atlas_model/`). Changing the state
  encoder's features means updating that copy too.
- **One game at a time per process.** Never put it in the rollout workers,
  which run many games side by side. `ml.evaluation.evaluate` is safe.

**First result (2026-10-01):** the overnight model (round 500) against Atlas,
8 fresh maps from both seats, Atlas pinned to that same model: **4 wins, 12
losses (25%)**, 10 minutes for 16 games. Few games, so the true score could
lie anywhere from about 8% to 50%, but Atlas is clearly the stronger player.
A 25% score corresponds to roughly 190 rating points below Atlas.

**Atlas uses copies of its opponents** (looked at 2026-10-01). Its forecasts of what the opponent will do come from three
sources:

| Source | Clean? | What it gives Atlas |
|---|---|---|
| The code of `strategic_ai`, `careful_ai`, `aggressive_ai` (imported as hypotheses) | No: an exact copy of the opponent | Once observed moves match one exactly, Atlas simulates that very opponent to the end of the match and keeps a "certified" winning line while the moves keep matching. Most of its 129–0 against the scripted AIs rests on this. |
| Our checkpoint (by default `ml/runs/latest.pt`, the network it is playing) | No: the opponent's own brain, minus its dice | Against a network, 60-second futures with the network predicting our moves |
| Its own ten policy styles (`CONFIGS`), fitted to the moves it sees | Yes: its own strategy, adapted to the board | A best-fitting style per opponent, plus a cautious second scenario when nothing fits, scored as 65% average and 35% worst case |

A clean Atlas keeps the simulator, the policy portfolio, the search, and the
fitting of observed moves, but fits only its own styles. `ais/_atlas_classic_ai.py`
is already most of the way there: it has no network branch, so only the three
scripted AIs would need removing from its hypotheses. Its strength without
the copies is unknown and has to be measured: the certified winning lines
against scripted AIs disappear, and against our network it loses the most
accurate forecast. The earlier controller without the network branch beat an
old imitation-era snapshot in all 16 duels, but that snapshot was far weaker
than round 500.

**Atlas Clean** (`ais/atlas_clean_ai.py`, built 2026-10-01; the original
Atlas is untouched):

- Forecasts every opponent with Atlas's own ten styles only, fitted to the
  launches it sees, plus the cautious second scenario when none fits. It
  imports no other AI and no network (tested), and it drops the "certified"
  winning lines, which only ever applied to the copied scripted AIs.
- Because no opponent is ever "familiar" any more, it always does Atlas's
  tick-by-tick refinement of its three best styles. Planning all at once, as
  Atlas does, that froze the game window once per game second: 0.5 s on
  `classic`, 1 s on an 8-base map, 2.2 s on a 24-base map (median planning
  tick, measured while a training run used the other cores).
- **Planning spread over the ticks** (`WORK_PER_TICK` = 120 simulation steps
  per decision, a count, not a time, so matches stay repeatable): 35–50 ms
  per decision at the median, 76–95 ms at the 99th percentile, 220 ms at
  worst, on 8–24 bases under the same load. The price: a plan is 2.5–4 game
  seconds old when it takes effect, against instantly before. Policy calls
  are most of the cost (median 0.1–0.15 ms, but up to 25 ms on a busy large
  position). The plan-at-once version stays available for comparisons as
  `_atlas_clean_blocking` (hidden from the menu). Still too slow for 96
  training games per round.
- **Memory per match:** `new_player()` returns a player with fresh memory.
  Our runners (`ml.play.rollout`, `ml.evaluation.evaluate`) give such an AI
  its own player in every game (`keeps_memory`, `fresh_ai`), so many games
  can run side by side in one process. Tested: two games on the same map and
  seat at once play exactly as each does alone; with one shared player they
  do not. The module's `choose_action` is one shared player for the game
  window, and keeps the seats of that one match apart.
- [x] Measured (`ml/scripts/done/atlas_clean_check.sh`, log in
      `ml/runs/logs/atlas_clean_check_log.txt`), 16 fresh maps from both seats, 32
      games per opponent, the original Atlas's network pinned to round 500:

| Atlas Clean against | Score | Wins / draws / losses |
|---|---|---|
| `careful_ai` | 93.8% | 30 / 0 / 2 |
| `aggressive_ai` | 90.6% | 29 / 0 / 3 |
| `strategic_ai` | 73.4% | 23 / 1 / 8 |
| overnight round 500 | 46.9% | 15 / 0 / 17 |
| itself, planning all at once | 37.5% | 11 / 2 / 19 |
| the original Atlas | 17.2% | 5 / 1 / 26 |

- **Without copies, Atlas is about as strong as our overnight network:**
  even against round 500, and the same score against `strategic_ai` that
  round 500 gets (about 73%). On our scale that is roughly 1150–1180.
- **The original Atlas is far stronger** (26 of 32). It does not recognise
  Atlas Clean either, so against it the original uses its network branch:
  a trained network forecasting the opponent, with about ten times the
  compute. So most of the original's edge here is a better forecaster, not
  an exact copy of this particular opponent.
- **Spreading the planning costs something:** 11–19 against its own
  plan-at-once version, about 1.5 standard errors from even, so likely
  real but not certain.
- **Below the move-scoring player:** that one beat round 500 64% to 36%,
  where Atlas Clean is even with it.

## League against plain self-play (from 2026-10-01)

**Question:** the overnight run (only ever against its current self)
stopped improving after about round 300. Does a league get it moving again?

**Design** (`ml/scripts/done/league_vs_selfplay.sh`): both arms start from
overnight round 500 and train 50 more rounds, two seeds each, with the same
settings: the step size the overnight run ended with (5.9e-5), no
value-only rounds, no material hint. Self-play plays only itself; the league
plays itself half the time, past champions 30%, scripted AIs 20%, plus a
128-game promotion match every 5 rounds (not trained on). Then `ml compare`
on fresh maps: rounds 0, 25, 50 of every run against `strategic_ai` and
overnight rounds 300, 400, 500, 100 games each, and the two arms' final
rounds head to head. Round 0 is the same model in all four runs, so its four
ratings show the measuring noise. About 2 hours.

- [x] Started 2026-10-01 17:02, right after the capacity probe (killed by a
      tool time limit during the third run at 18:05, resumed, finished 19:25)
- [x] Result, below
- [x] Decide: plain self-play for the long run

**Result (2026-10-01)**, report in `ml/runs/compare_league_vs_selfplay/report.txt`
(`strategic_ai` fixed at 1000; each checkpoint rated from 400 games):

| Round | League, mean of 2 seeds (range) | Self-play, mean of 2 seeds (range) | Score vs `strategic_ai`, league / self-play |
|---|---|---|---|
| 0 (= overnight round 500) | 1163 (1142–1183) | 1158 | 71% / 72% |
| 25 | 1174 (1137–1212) | 1191 (1188–1194) | 75% / 73% |
| 50 | 1175 (1156–1194) | 1182 (1175–1189) | 74% / 74% |

- **Neither moved off the plateau, and they cannot be told apart.** Both
  gained 10–20 points in 50 rounds, inside the noise. Head to head, the final
  self-play models scored 51.5% against the final league models over 400
  games (0.6 standard errors from even).
- **The noise, measured:** round 0 is the same model in every run, and it
  was rated anywhere from 1142 to 1183 (the panel's copy of it: 1164). Two
  ratings from 400 games each need to differ by more than about 40 points
  before they mean anything.
- **The league rarely promoted:** 3 promotions in 20 matches, each barely
  over 55%; most matches against the champion ended 42–54%.
- **What this says about the plateau:** it is not caused by playing only
  against itself, at least not over 50 rounds. With size not being the main
  limit either (see "Network size"), the remaining suspects are what the
  network is shown, the training signal, and the lack of look-ahead.
- **Caveat:** the league's past champions were all close copies of round
  500, so its pool had little variety. A league grown from scratch over a
  long run meets more varied opponents; that was not tested.
- **Decision:** plain self-play for the long run. It is as good, simpler,
  and what the tuning was done on. The league stays available.

## The deeper network's overnight run (2026-10-01/02)

`deep_scratch`: 64 wide, 6 layers (220,932 weights), 500 rounds of plain
self-play from random weights, settings as the first overnight run. 5.6
hours (the first took 3.8). Rated on one scale with the first overnight run
(`ml/runs/compare_deep_vs_overnight/report.txt`; `strategic_ai` = 1000,
100 games per pairing):

| Round | 64×3 (first overnight) | 64×6 (deep) | vs `strategic_ai`, 64×3 / 64×6 |
|---|---|---|---|
| 0 | 394 | 451 | 2% / 4% |
| 50 | 840 | 870 | 30% / 36% |
| 100 | 942 | 1048 | 44% / 59% |
| 200 | 1070 | 1139 | 60% / 70% |
| 300 | 1143 | 1225 | 68% / 74% |
| 400 | 1145 | 1242 | 66% / 80% |
| 500 | 1175 | 1262 | 76% / 79% |

- **The deeper network is stronger at every point from round 100 on**, by
  about 80–100 points at the end. Head to head, the final deep model won
  61.5 of 100 games against the final 64×3 model (2.3 standard errors).
- **Also per hour of training:** at 3.8 hours, where the first run ended
  at 1175, the deep run was at about round 340 and about 1230.
- **Still slowly improving:** +17 and +20 points per 100 rounds after round
  300 (64×3: +2 and +30, noisier). The plateau is higher, not gone.
- **Caveats:** one seed each; single checkpoints rated from about 200 games
  are good to about ± 30 points, but the gap holds at every round.
- The imitation start for the same network: see Stage 10.

## Two-move look-ahead (2026-10-02)

**Built** (`ml/play/lookahead.py`, player name `lookahead:PATH`): our 8 most
likely moves plus waiting, against the opponent's 8 most likely plus
waiting (as our own network sees them from the opponent's side); every pair
judged with both fleets on their way, a table of 81 positions in one batch;
our move chosen from the table by `rule`: `worst` (best worst case),
`expected` (best average over the opponent's likely moves), or `nash` (the
mix of our moves no mix of theirs can exploit, solved by regret matching;
the move it uses most). 40–60 ms per decision on the deep network.

**Stage 1** (`ml/scripts/done/lookahead_check.sh`, log `ml/runs/logs/lookahead_check_stage1_log.txt`),
all on `deep_from_imitation` round 500, 60 games per pairing on 30 fresh maps:

| Rule | vs plain network | vs move scoring | Overall |
|---|---|---|---|
| `nash` | 60.8% (33–20, 7 draws) | 60.0% (33–21, 6 draws) | 60.4% |
| `expected` | 57.5% (32–23, 5 draws) | 60.0% (32–20, 8 draws) | 58.8% |
| `worst` | 59.2% (33–22, 5 draws) | 54.2% (30–25, 5 draws) | 56.7% |

- All three rules beat both the plain network and one-move scoring; `nash`
  by the most. At 60 games each, about 1.5 standard errors: likely, not
  proven, and the rules cannot be told apart from each other.
- Note: move scoring's 8 candidates are now always 8 different moves (before,
  two amount options with the same units could leave fewer).
- [x] Stage 2 (log `ml/runs/logs/lookahead_check_stage2_log.txt`): `nash`
      against `strategic_ai` and the original Atlas (its opponent network
      pinned to the same deep network), 60 games each on 30 fresh maps:

| Against | Score | Wins / draws / losses |
|---|---|---|
| `strategic_ai` | 90.0% | 53 / 2 / 5 |
| the original Atlas | **66.7%** | **38 / 4 / 18** |

- **The strongest player so far, and clearly ahead of Atlas:** 38–18 is
  about 2.7 standard errors from even. (Move scoring on the deep
  random-start model had beaten Atlas 23–14, about 1.5 standard errors.)
- It costs 40–60 ms per decision, against Atlas's 220 ms against a network.

## The judge: a separate value network (planned 2026-10-02)

**Why:** in the traced loss (see "Losses to the weak AIs"), the look-ahead's
judge, the value output of `deep_from_imitation`, decided between expanding
and shuffling on gaps of 0.04–0.08 that are noise. Today's value output
learns from one game result per position, discounted by the time left, on
96 games per round, sharing the network with the policy, and its updates
stop when the policy has moved far enough. A judge of small differences
needs many more positions, a target without the "when", and its own weights.

**Why a judge from our own games can be good enough:** it learns "how often
does this player win from here". That is a property of the player, not
perfect play, but the main signs of a good position (production, units,
threats, cheap captures) hold for any reasonable player, and picking the
move whose resulting position the judge likes best plays at least as well
as the player it learned from (the policy improvement theorem; exact for one
player, approximately true with an opponent). That is what move scoring
and the look-ahead already showed. A better player then gives a better
judge, which is the loop below. AlphaGo trained its value network on its own
policy's games; AlphaGo Zero started from random play.

**The steps:**

1. **The judge network.** Input: a position from one player's point of
   view (the existing encoder, whichever input set the input check picks).
   Output: the probabilities of win, draw, and loss, adding up to 1,
   learned with cross-entropy against the actual result, without discount.
   As the look-ahead's judge it gives the expected score, P(win) + ½·P(draw).
   Same graph transformer, value part only. The look-ahead keeps using
   `deep_from_imitation`'s policy to propose the candidate moves; only the
   judging changes.
2. **The games.** Every position of a game is a data point, labelled with
   that player's final result; every seat is recorded, also the scripted
   AIs' seats, about 50 positions per seat per game (neighbouring ticks are
   nearly the same). First round: about 10,000 games of the plain
   `deep_from_imitation` against itself (it picks by chance, so games
   differ), plus about 500 each against `aggressive_ai`, `careful_ai`,
   `strategic_ai`, and older checkpoints, so the judge also knows how games
   against other styles go. Optionally about 1,000 look-ahead games on top
   (it is about 40 times slower, and plays the same game on a given map).
3. **Training.** Supervised, a tenth of the games held out (by game, not by
   position), then a test set of fresh games. Watch the held-out loss and the
   calibration: positions it calls 70% should win about 70%.
4. **The test.** The look-ahead with the new judge against the look-ahead
   with the old one (head to head), against the original Atlas and the plain
   network (fixed references), and the saved weak-AI losses again
   (`ml rematch`): does it expand where it used to shuffle?
5. **New baseline:** if it wins, the look-ahead with the new judge is the
   strongest player, and it plays the next round's games.
6. **Repeat 2–5** while it pays. The loop improves only the judge; the
   policy that proposes the candidates stays fixed, so gains will level off.
   Then the policy has to learn as well (learning from the look-ahead with
   soft targets, or PPO with the better value output).

**Progress you can watch** (your wish): one block per phase, updated in
place in a terminal and as lines in the log.

```
=== Generating games ===
(5,243 / 10,000) vs itself          (12m remaining)
(    0 /    500) vs aggressive_ai
(    0 /    500) vs careful_ai
(    0 /    500) vs strategic_ai
(    0 /    500) vs older checkpoints
positions recorded: 524,300 (training 471,870 / held out 52,430)

=== Training judge ===
epoch  3/20   train loss 0.612   held-out loss 0.634   (2m10s per epoch)
...
test loss 0.641   calibration: said 70% -> won 68%

=== Evaluation ===
head to head vs old judge (112 / 200):  61 wins / 9 draws / 42 losses
vs Atlas           (  0 / 60)
vs plain network   (  0 / 60)
rematches of saved losses (3 / 10): expanded in 2
```

For the whole loop, the same blocks under an iteration header
(`=== Iteration 2 of 5 ===`), plus a one-line summary per finished
iteration (judge's test loss, head-to-head score, score vs Atlas), and the
numbers in a `metrics.jsonl` so `ml monitor` can chart the loop.

**Where it runs:** game generation is single-threaded Python per game and
is most of the time; it runs best on the laptop's cores. The training step
could run on a Colab GPU (see "Open questions"), but at these sizes the
laptop's GPU should take minutes, not hours.

- [x] Steps 1–3 built (2026-10-02): the judge network (`ml/models/judge.py`,
      model name `graph_judge`: the playing network's attention layers, the
      average and the largest of the base descriptions, three outputs), data
      generation (`ml judge-games`, `ml/judge/games.py`, resumable with
      `--resume`), training (`ml judge-train`, `ml/judge/train.py`;
      `--start-from` copies a playing network's attention layers in; the
      test report gives test loss, score error next to the playing network's
      value output, and calibration). `lookahead:` and `value:` players take
      `&judge=PATH`. Tests in `ml.tests.test_judge`.
- [x] Step 4 built: `ml judge-check --judge PATH` (`ml/judge/check.py`)
- [x] Progress as you asked (2026-10-02): the groups are played one after
      the other, every finished game updates the screen (the rollout pool
      now reports games as they end), and each group's line keeps its own
      time: "took 12m30s" when done, the time it still needs while it runs.
- [x] First round of games (`round1`, 2026-10-02): 12,500 games, 499,772
      positions, 42m20s of playing, 1,046 MB (2.1 KB per position); the
      10,000 games against itself took 35m42s.
- [x] Training shows what it is doing: a live line per batch (bar,
      train loss so far, time left), a line per epoch, the baseline
      explained in words. About 55 seconds per epoch on round1.
- [x] First real judge (`judge1`, 20 epochs, started from the playing
      network's attention layers): **it overfits after epoch 6.**

| Epoch | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| training loss | 0.413 | 0.360 | 0.328 | 0.307 | 0.284 | 0.255 | 0.225 | 0.197 | 0.174 | 0.158 | 0.151 |
| held-out loss | 0.368 | 0.346 | 0.345 | **0.334** | 0.354 | 0.352 | 0.383 | 0.410 | 0.445 | 0.464 | 0.473 |

  - The held-out loss is lowest at epoch 6, then climbs every epoch while
    the training loss keeps falling: from there on it learns the training
    games themselves, not positions in general. `best.pt` is epoch 6.
  - **Epoch 6 on the test games:** loss 0.349 (baseline 0.989); score error
    0.0711 against 0.0888 for the playing network's value output, 20% lower.
    Calibrated to within about 2 points, except slightly too cautious at
    60–80% (said 65% won 70%; said 75% won 81%).
  - **Why it overfits:** 400,000 positions, but only about 10,000 games, and
    every game gives 40 positions that share one result and look much alike.
    A 212,000-weight network can learn to recognise the game and recall its
    result. The number of games, not of positions, is what limits it. Also,
    55% of positions are already decided (under 10% or over 90%); the
    useful middle is a smaller set.
  - **The check** (`ml judge-check --judge ml/runs/judge1/best.pt`, 2026-10-02,
    `ml/runs/judge1/check.json`): the look-ahead (`nash`, policy from
    `deep_from_imitation` round 500) with the new judge, on the same fresh
    maps as the earlier look-ahead checks:

| Test | New judge | Old judge (earlier, same maps) |
|---|---|---|
| head to head, new against old (100 games) | **59.5%** (54–35, 11 draws) | |
| vs the original Atlas (60 games) | **82.5%** (48–9, 3 draws) | 66.7% (38–18, 4 draws) |
| vs the plain network (60 games) | 65.0% (35–17, 8 draws) | 60.8% (33–20, 7 draws) |
| rematches of the 37 saved losses | **25 won** | 17 won |

  - **Better on every test.** Head to head about 1.9 standard errors; against
    Atlas 16 points up (about 2 standard errors on the difference); 8 more
    of the saved lost starts won. Against the plain network the gain (4
    points) is within the noise.
  - **The strongest player so far:** the look-ahead with `judge1`, 82.5%
    against the original Atlas. It is **Beast AI** in the game menu
    (`ais/beast_ai.py`): Beast is always our strongest player, chosen in
    `ml/runs/beast.json`, shown and moved with `ml beast` / `ml beast set
    NAME --why ...` after any evaluation that finds a stronger one. About
    18 ms per tick against Neural AI; the first move of a match loads the
    networks (under a second).
- [x] Which saved losses `judge1` still loses (each distinct start once,
      22 starts; the saved losses list had duplicates of the same start):
      7. Against `aggressive_ai` only `900094` from red, a seat every player
      tested loses. It now wins `900064` from red (the shuffling game) and
      `900075` from red against `aggressive_ai`, which even `strategic_ai`
      lost. The other 6 are against `careful_ai` (`900075` and `900094` from
      red, `900065` both seats, `900040`, `900116`). `ml judge-check` now saves
      such games, and plays each distinct start once.
- [ ] **Later: do the other opponents' games help or hurt the judge?** The
      judge sees only the position, not who plays; against `aggressive_ai`
      a position may be a likely win because the opponent will blunder,
      while the look-ahead assumes an opponent as strong as itself. Measure
      with existing data, no new games: score the judge separately on the
      test positions from self-play and from the other opponents' games
      (loss, calibration), and against a judge trained on self-play alone.
      If the mix hurts: mostly self-play of the strongest player (85–90%),
      strong opponents of other styles (older checkpoints, Atlas Clean),
      a little `careful_ai` while we still lose to it, and only our own
      seats recorded in games against weak opponents.
- [x] Against overfitting (2026-10-02): judge training now has weight decay
      (0.01, on weights only, AdamW) and early stopping (stops once the
      held-out loss has not improved for 4 epochs; the best epoch is kept),
      and takes several data runs at once (`--data round1 round2`). A new
      data run refuses a seed another run already used, since the seed picks
      the maps and shared maps would let test games leak into training.
- [x] Round 2 and `judge2` (2026-10-02, `ml/scripts/done/after_round2.sh`, log
      `ml/runs/logs/after_round2_log.txt`): round2 = 37,500 games, 8 positions
      per seat. `judge2` on round1 + round2 (881,774 training positions,
      linear unit counts, random start): best epoch 7 of 11; test loss 0.311
      (round-1 judges about 0.34), score error 0.0681 against the playing
      network's 0.0876 on the same positions (22% lower; `judge1` was 20%
      lower). Too optimistic in the middle: said 60–80%, scored 5–7 points
      less. In play (look-ahead `nash`, same network):

| Test | `judge2` | `judge1` (earlier, same maps) |
|---|---|---|
| head to head vs the old judge (100) | 58.5% (57–40, 3 draws) | 59.5% |
| **vs Beast, i.e. the look-ahead with `judge1` (100)** | **53.5% (50–43, 7 draws)** | |
| vs Atlas (60) | 80.8% (47–10, 3 draws) | 82.5% |
| vs the plain network (60) | 75.8% (42–11, 7 draws) | 65.0% |
| distinct saved lost starts (22) | 14 won | 15 won |

  - **About as strong as `judge1`, not clearly stronger.** 53.5% against
    Beast is 0.7 standard errors from even; Atlas and the rematches are level;
    only against the plain network is it clearly better. Beast stays as it is.
  - More games made the judge more precise on paper (score error down 3%),
    but that hardly shows in play.
- [x] Old command, for the record: `ml judge-games --name round2 --seed 1 --self-games 30000 --games-per-opponent 1500 --positions-per-seat 8`
      (about 2 hours, about 600,000 positions), then
      `ml judge-train --data round1 round2 --run judge2 --start-from ml/runs/deep_from_imitation/round_0500.pt`
      and `ml judge-check --judge ml/runs/judge2/best.pt`
  - **What to do:** use `best.pt` (epoch 6) for the check now. For the next
    round: more games and fewer positions per game (e.g. 30,000 games at 8
    per seat: the same size on disk, three times the results), and in
    training weight decay and a schedule that ends where it stops improving
    (about 8 epochs) instead of 20.
- [ ] The loop (steps 2–5 repeated), with per-iteration summaries

## Units: log scale or plain counts? (2026-10-02)

Your view: every quantity once, and on the scale the game uses; fights are
subtractions, a capture needs a fixed number of units, so relations
(ratios) do not matter here. The log scale was chosen for the range of
unit counts (0 to over 1,000), not for ratios.

**The test** (`ml/scripts/done/judge_units_check.sh`): two judges on round1,
same settings (weight decay, early stopping), both from random weights,
differing only in how they read unit counts (units, incoming fleets, units
once fleets land, a capture's cost): log, as the encoder gives them, or
plain counts / 250 (`--set unit_scale=linear`; the judge turns the log
values back into exact counts itself, so no new games are needed and the
judge can be used in play as it is).

**Result (2026-10-02):** no difference.

| | log (`judge_log`) | linear (`judge_linear`) |
|---|---|---|
| best epoch (of 14, both stopped early) | 10 | 10 |
| held-out loss at the best epoch | 0.332 | 0.329 |
| test loss | 0.338 | 0.345 |
| test score error | 0.0705 | 0.0707 |
| for comparison: `judge1` (no weight decay, started from the playing network) | test loss 0.349, score error 0.0711 | |

- The two are within the noise of each other: the held-out loss of one
  run moves by 0.01–0.02 from epoch to epoch, more than the gap between
  them. Calibration is equally good, off in opposite directions in the
  10–40% band (log too low, linear too high, by 3–6 points).
- So the scale of unit counts is not what limits the judge. Since it does
  not matter, the cleaner choice is fine: linear, every count once, in the
  game's own terms. New judges can use `unit_scale=linear` at no cost.
- Weight decay and starting from random weights did not change much either
  (0.0705 against `judge1`'s 0.0711). What limits the judge is most likely
  the number of independent games (see "The judge"): round 2 is the lever.

## Teaching the policy from the look-ahead (step 2 of the loop, built 2026-10-02)

The loop: a better judge, then a better policy, then better games, then a
better judge. This is the "better policy" step. The earlier attempt
(distilling move scoring, see "Move scoring") failed because the network
copied a single best move and was pulled into a different way of playing.

**A. Targets** (`ml policy-targets`, `ml/judge/targets.py`): the network
plays games on instinct (against itself, a few against other AIs); in about
10 positions per seat, the look-ahead (the same network proposing, the judge
judging) works out its table. Kept per position: the network's own
probability of each candidate and the look-ahead's score for it (what it
earns against the opponent's equilibrium mix of replies). No look-ahead
games are needed: about 4 seconds of one core per game.

**B. Learning** (`ml policy-train`, `ml/judge/policy.py`): the target among
the candidates is the network's own probability times exp(score / tau),
renormalised to the probability the candidates had together; all other
moves keep their share. So the step reshuffles weight among the candidates
towards the better-judged ones instead of squeezing everything onto them.
Loss: cross-entropy to that target, plus the KL divergence from the starting
network over all moves (`kl_weight`), plus the value error. Early stopping
on the held-out positions. Progress shown per batch; per epoch the "gap to
the best candidate": how much worse, by the look-ahead's judgement, the
network's choice among the candidates is than the best one.

**C. Check** (`ml policy-check`): the taught network on instinct against the
network it was taught from, the look-ahead with the taught network against
Beast, and against the original Atlas.

**First observation (small smoke test, 920 positions):** the judge's scores
for the candidates of one position differ very little: median 0.001 between
the best and the second best, 0.012 between the best and the worst (scale −1
to +1). So tau is 0.01, and the signal per position is small: it needs many
positions, and success has to be measured by play, not by agreement.

- [x] Built, with tests (`ml.tests.test_teaching`)
- [x] First run (2026-10-02, part of `ml/scripts/done/after_round2.sh`), taught
      with `judge2`: 2,800 games in 18 minutes, 52,997 positions; training
      stopped at its best epoch (16 of 19).

| Test | Result |
|---|---|
| the taught network (`policy1`) on instinct vs the original (100 games) | 51.5% (45–42, 13 draws) |
| look-ahead with `policy1` vs Beast (99 games; the 100th was lost in a crash) | 49.5% (47–48, 4 draws) |
| look-ahead with `policy1` vs Atlas (60 games) | 85.0% (50–8, 2 draws); Beast had 82.5% |

  - **No measurable effect, neither better nor worse.** Every result is
    within the noise of the network it was taught from.
  - The gap to the best candidate shrank only from 0.0150 to 0.0135 (10%),
    while the drift (KL from the start) was 0.16 after the first epoch and
    0.17 at the end: as much change as about ten self-play rounds, with no
    change in strength.
  - **A measuring flaw that likely explains the drift:** when two amount
    options send the same units (e.g. "just enough" and an eighth), the
    targets merge them into one move, but training raises only one of the
    two options. Training then moves probability between options that play
    identically: a large KL, no change in play. Moves, not options, should
    be compared.
  - **The real signal is small:** the judge's scores for one position's
    candidates differ by about 0.01, near what the judge can tell apart.
  - **Crash:** the Atlas part of `policy-check` failed: Atlas's frozen copy
    of the encoders refuses networks saved since the inputs grew. Fixed:
    Atlas now forecasts with the network the policy was taught from. The
    Atlas games were then run on their own (`ml/runs/logs/policy1_atlas_log.txt`).
- [x] Next for teaching, cheapest first: compare moves instead of options in
      targets, loss, and drift; a stronger step (smaller tau); then the
      judge as the critic in self-play instead (more precise credit for
      real games), or a deeper look-ahead whose candidates differ more.
      → done the night of 2026-10-03, see below.

## Night of 2026-10-03: moves, not options; the judge as critic

The plan for the night (about 8 hours, unattended):

1. **Fix the teaching flaw.** The targets now record, per candidate, which
   amount options send exactly its units (`same_units`, one bit per
   option); the loss and the "gap" count the network's probability of a
   move as those options together. The drift still compares options, which
   is now harmless: nothing pulls probability between equivalent options
   any more, so holding their split where it was costs nothing. New
   targets `teach2` (seed 101, same settings as `teach1`; `teach1` lacks
   the record). Retrain at tau 0.01 and 0.003; quick check on instinct
   (200 games against the original); only a network at 55% or better gets
   the full look-ahead check.
2. **The judge as critic** (`--critic PATH` in self-play, any trainer): a
   move's advantage compares how things went with what the frozen judge
   expected, instead of the network's own value output, which learns from
   one round's games and moves with every round. The network's value
   output still learns, and is still reported; the critic's error is
   reported next to it. Cost: about 8 s per round for ~150,000 positions.
   Small known bias: the judge predicts the undiscounted result, while the
   advantages discount by 0.999 per tick; tolerated, as it is small next
   to the noise.
3. **A/B** (`ml/scripts/done/critic_ab.sh`): both arms continue
   `deep_from_imitation` from round 500, 150 rounds each, same seed and
   settings (step size 4.4e-5, no value-only rounds, no material hint);
   arm `ab_plain` as before, arm `ab_critic` with `judge2`. Then
   `ml compare` of rounds 0/50/100/150.
4. The winner's last round as the look-ahead's network (with `judge2`)
   against Beast (100 games) and Atlas (60); Beast updated if clearly
   better.

- [x] Built, with tests (`CriticTests` in `ml.tests.test_selfplay`,
      `CandidateProbabilityTests` in `ml.tests.test_teaching`)

**Teaching, moves instead of options** (`ml/scripts/done/teach2.sh`; targets
`teach2`: 2,800 games, 52,992 positions, 34 minutes next to the A/B run):

| | tau 0.01 (`policy2_tau01`) | tau 0.003 (`policy2_tau003`) | before: `policy1`, tau 0.01, options |
|---|---|---|---|
| best epoch | 7 | 2 | 16 |
| drift (KL from the start), held out | 0.036 | 0.072 | 0.17 |
| gap to the best candidate (was 0.0164) | 0.0152 | 0.0152 | 0.0135 (was 0.0150) |
| on instinct vs the original, 200 games | **49.8%** (88–89, 23 draws) | **44.0%** (80–104, 16 draws) | 51.5% (100 games) |

- **The flaw was real:** comparing moves cut the drift by a factor of
  five at the same tau. Most of `policy1`'s change was probability moved
  between options that play identically.
- **But teaching still does not help:** at tau 0.01 no change in play; the
  stronger step (tau 0.003) is worse (44%, about 1.7 standard errors below
  even). Neither reached 55%, so neither got the look-ahead check.
- **Why, most likely:** the network barely learns to predict which
  candidate the judge prefers. The held-out target loss falls only from
  1.239 to 1.225 and the gap by 7%: the judge's preferences among one
  position's candidates (differences of about 0.01) are mostly not
  learnable from the position, i.e. close to noise. A stronger step then
  only copies noise. This closes "teach from the one-step look-ahead with
  this judge"; it would need candidates whose values really differ (deeper
  look-ahead, or a judge that separates them more clearly).

**A/B, plain against frozen critic** (`ml/scripts/done/critic_ab.sh`). In-run
checks (16 games each, so only rough):

| round | plain: vs start / strategic | frozen critic: vs start / strategic | critic's error (always 0) |
|---|---|---|---|
| 10 | 50% / 100% | 44% / 88% | 0.14 (0.23) |
| 30 | 62% / 81% | 34% / 41% | 0.32 (0.15) |
| 50 | 38% / 88% | 6% / 34% | 0.29 (0.14) |
| 100 | 44% / 88% | 0% / 3% | 0.19 (0.15) |
| 150 | 38% / 94% | | |

- **The frozen critic was fooled.** From about round 20, the network
  learned to steer into positions the judge misjudges in its favour: the
  judge's error against how the games ended rose from 0.14 to 0.33, worse
  than always saying 0, while the network fell to 0% against its starting
  point. Games got longer (100 → 150 s) and more were left undecided; the
  policy's entropy fell from 3.4 to about 2. This is the classic way a
  fixed learned reward goes wrong: optimising against it finds its
  mistakes. (The network's own value output, which learns every round,
  stayed accurate: error 0.04–0.08.)
- Part of the critic's error is a mismatch, not a mistake: the value
  target is discounted (0.999 per tick) while the judge predicts the plain
  result. That explains its starting 0.14 against the network's 0.085, but
  not the rise.
- **Next (started straight away, `ml/scripts/done/critic_learning.sh`):** the
  same, but the critic keeps learning (`--critic-learning-rate 1e-4`: one
  pass per round over a sample of that round's positions, towards how the
  games ended; saved as `critic.pt` in the run). It starts with judge2's
  knowledge, but a blind spot the policy finds gets corrected. Control:
  `ab_plain`.

**Rating comparison** (`ml/runs/compare_critic_ab/`, 100 games per match,
strategic_ai = 1000; round 0 is the same network in both arms, so its two
ratings, 1300 and 1313, show the measuring noise):

| round | plain | frozen critic |
|---|---|---|
| 0 | 1300 | 1313 |
| 50 | 1242 | 1071 |
| 100 | 1309 | 776 |
| 150 | **1353** | 1001 |

Last rounds head to head: plain 85.5% (+7.1 standard errors). Plain self-play
still creeps up (+50 in 150 rounds, from a dip at round 50); the frozen
critic lost up to 530 points and had recovered half by round 150.

- [x] Look-ahead check of `ab_plain` round 150 (with judge2), next to the
      learning-critic run (`ml/runs/ab_plain/check.json`):

| Test | Result |
|---|---|
| on instinct vs deep_from_imitation round 500 (100 games) | 54.0% (50–42, 8 draws) |
| look-ahead vs Beast (100 games) | 51.0% (50–48, 2 draws) |
| look-ahead vs Atlas (60 games) | 75.0% (43–13, 4 draws); Beast had 82.5% |

  Not clearly stronger, so **Beast stays** (look-ahead with
  deep_from_imitation round 500 and judge1). 150 more rounds of plain
  self-play buy about +50 rating points on instinct, which the look-ahead
  does not turn into a measurable gain.

**Learning critic** (`ml/scripts/done/critic_learning.sh`, run
`ab_critic_learning`; comparison `ml/runs/compare_critic_learning/`):

| round | plain | learning critic | critic's error / network's / always 0 | game length |
|---|---|---|---|---|
| 0 | 1291 | 1289 | 0.14 / 0.085 / 0.31 | 103 s |
| 50 | 1234 | 1254 | 0.13 / 0.078 / 0.22 | 129 s |
| 100 | 1300 | 1178 | 0.09 / 0.057 / 0.17 | 140 s |
| 150 | 1343 | 1155 | 0.11 / 0.044 / 0.18 | 144 s |

Last rounds head to head: plain 72.5% (+4.5 standard errors).

- **No collapse, but still worse than plain:** the critic kept its error
  in check (it improved, 0.14 → 0.09–0.11), but the network slowly lost
  about 130 points while plain gained 50.
- **Games got longer** (103 → 144 s, plain stayed at about 90–110 s). A
  likely cause, not tested: the judge predicts *who wins*, not *how soon*.
  The advantages discount by 0.999 per tick, but a critic that does not
  discount gives no extra credit for winning sooner, so nothing pushes the
  network to finish games. The network's own value output is trained on
  the discounted result and does push.
- **Conclusion for now:** the judge helps the look-ahead (Beast) but not
  self-play as a critic, in either form. If tried again: a critic that
  predicts the discounted result (fine-tuned with that target), so
  speed counts.

**Then (started 08:08): plain self-play carried on**, `ab_plain` from
round 150 to 600 (`ml/scripts/done/plain_long.sh`; settings.json changed to 600
rounds, nothing else). It is the one method still improving the network.
  Stopped by request after round 400 (11:05), which is in effect round
  900 of deep_from_imitation (fresh optimizer state at round 500).
- [x] Evaluation, run by hand: `ml/scripts/done/plain_400_check.sh`. `ml
      compare` now shows a live board of every match while it plays. The
      look-ahead part was cancelled (it would not have changed a decision).

| ab_plain round (deep_from_imitation round) | rating | vs strategic_ai |
|---|---|---|
| 0 (500) | 1296 | 82% |
| 150 (650) | 1320 | 88% |
| 250 (750) | 1357 | 90% |
| 400 (900) | 1345 | 88% |

Panel: strategic_ai 1000, aggressive_ai 715, deep_from_imitation round 400
1215, deep_scratch round 300 (the deep network trained without imitation)
1142. Plain self-play still adds about +50 over 400 rounds, and is
flattening out (250 and 400 are within the noise of each other).

## A larger network: throughput, tactical amounts, capture prediction (2026-10-03)

The related-projects research (`docs/related_projects.md`) points to larger
networks trained on far more games. Our network has 0.22 million weights
and about 0.14 billion self-play steps behind it; the leading Orbit Wars
networks have 4-200 million weights and 2-15 billion steps. Plan agreed
with the user: (1) batched GPU inference, measure the round time; (2) add
tactical amounts and a capture prediction; (3) warm-start a larger network
from ours and compare it with ours.

**1. Where a round's time goes** (96 games, 6 workers, M2 Pro):

| network | playing (CPU workers) | learning (GPU) |
|---|---|---|
| 64x6 (0.22 M weights) | 22 s | 26 s |
| 128x8 (1.1 M) | 44 s | 36 s |
| 256x8 (4.5 M) | 88 s | 65 s |

Playing is mostly the Python game simulation and encoding; the network is
about half of it at 64x6. Learning (4 passes over ~37,000 positions) is
real network work on the GPU, not batch preparation (0.3 s per pass).

**GPU inference server** (`ml/play/inference.py`, `RolloutPool(...,
inference_device="mps")`, self-play setting `--inference server`): workers
simulate and encode, write positions into shared memory; one server in the
main process answers all waiting positions in one GPU batch (fixed shapes,
moves drawn on the GPU, small batches on the CPU instead). Each worker's
games are split in two groups that take turns. Correct (recorded
probabilities match the networks to 3e-5; local play is bit-identical to
before), but **no faster on this machine**:

| round | network | CPU workers | GPU server |
|---|---|---|---|
| 96 games | 64x6 | 21.9 s | 42.5 s |
| 96 games | 128x8 | 43.5 s | 70.3 s |
| 288 games | 64x6 | 55.6 s | 59.9 s |
| 288 games | 128x8 | 143.8 s | 148.0 s |

Why: a round is a chain of 1,800 ticks (the longest game), each needing an
answer before the next. The M2 Pro GPU costs at least ~4 ms per call
however small, and under the load of six busy workers it handles only
about 10,000 positions per second of the 128x8 network (24 ms per batch of
~250), about what the six CPU cores manage themselves. bfloat16 and
float16 crash in Apple's matrix kernels; `torch.compile` saves 10-30%.
So: kept as an option (it would pay off on a CUDA GPU or with much larger
networks), not used. A larger network is affordable anyway: 128x8 costs
about 1.6 times as much per round as 64x6 (80 s against 48 s).

Found on the way, not changed (it affects every run equally): learning
batches are formed per map size and filled up to 1,024 with repeats, so
positions from rare map sizes count up to ~4 times and about a quarter of
all learning rows are repeats.

**2. Tactical amounts and capture prediction**
- Two new amount options, appended (older networks keep reading the first
  9 and never pick the new ones; old trajectories are re-packed on read):
  - **hold**: the smallest fleet that takes the target and still holds it
    after every fleet already on its way there has landed; for an own base
    that would fall, the reinforcement it needs. Offered where it differs
    from "just enough". (One option covers both "capture and hold" and
    "reinforce": the same calculation.)
  - **surplus**: everything the source can send and still hold out against
    every fleet on its way to it; offered only when that is less than all.
  - Tested by playing positions forward: "hold" always kept the target, one
    unit fewer never did; "surplus" always kept the source, one more did
    not. Cost: `legal_actions` 37 → 56 µs per position (+8% simulation).
- **Capture output** (`capture_head`): for each route and amount, "is the
  target ours right after this fleet lands?". Every recorded send gets the
  answer from the game itself as its fleet lands (`Trajectory.captured`;
  checked against replays). Learned with weight 0.1 next to the moves and
  value (`--capture-weight`).

**3. Kickstart** (`ml train kickstart`, `ml/training/kickstart.py`): the new
network learns the teacher's probability for every move (KL divergence,
not just its chosen move), its value, and the capture labels, from 1,200
of the teacher's games. The new options get 5% of each route's amount
probability where on offer, so self-play can find out whether they help.

**Experiment** (`ml/scripts/done/large_network.sh`, started 13:01): 128x8 with
both additions, kickstarted from ab_plain round 400; then 150 rounds of
self-play against the same 150 rounds for the 64x6 teacher itself (same
settings as ab_plain, 96 games per round, CPU workers); `ml compare` of
rounds 0/50/100/150 on the plain_400 panel.
- [x] Kickstart (`ml/runs/large_kickstart/`): 1,200 teacher games in 4
      minutes, 213,639 positions to learn from (23,854 held out); 53 s per
      pass; stopped after epoch 15, best epoch 12 (17 minutes).

| held out | before | epoch 1 | epoch 12 (best) |
|---|---|---|---|
| divergence from the teacher | 1.147 | 0.533 | 0.375 |
| same favourite first step | 8% | 51% | 62% |
| value error vs the teacher | 0.329 | 0.017 | 0.005 |
| capture error (50:50: 0.693) | 0.679 | 0.168 | 0.141 |

  On instinct against its teacher (200 games): **44.8%** (83–104, 13
  draws), about 35 rating points below it. A close but imperfect copy:
  after epoch 12 the held-out divergence rose again while the training
  one kept falling, so more teacher games would have helped more than
  more passes; the 5% given to the new options also costs a little until
  self-play sorts them out. The large arm therefore starts slightly behind
  the control; the comparison's round 0 shows by how much.
- [x] Large arm: 150 rounds in 2 h 43 min (65 s per round; the control
      needs about 48 s). Capture error during self-play 0.13.
- [x] Control arm (150 rounds, 1 h 51 min) and comparison
      (`ml/runs/compare_large_network/`, plain_400 panel):

| round | large (128x8, new options, capture) | small (64x6, control) |
|---|---|---|
| 0 | 1327 | 1344 |
| 50 | 1356 | 1335 |
| 100 | 1341 | 1305 |
| 150 | 1331 | 1319 |

  Last rounds head to head: large 49.5% (−0.1 standard errors). Panel:
  strategic_ai 1000, aggressive_ai 755, deep_from_imitation round 400
  1226, deep_scratch round 300 1133.
- **No gain from the larger network** (with the new options and the
  capture output) in 150 rounds; it made up the kickstart's small handicap
  and then stayed level with the control. **Both arms are flat**: plain
  self-play from round 900 no longer improves (1344 → 1319, within the
  noise), as the plain_400 comparison already suggested. When a five-times
  larger network plateaus at the same level, the limit is not the
  network's size but what it is taught; the nearly random amounts (below)
  point at the learning signal. Beast stays as it is.

**Do the new options get used?** (48 self-play games per network, fresh
maps; `scratchpad/option_usage.py`, to become a command if useful)

| large network | "hold" offered (of ~60,000 sends) | taken | "surplus" offered | taken |
|---|---|---|---|---|
| round 0 | 295 | 4.7% | 1,407 | 5.1% |
| round 50 | 270 | 3.3% | 1,485 | 6.5% |
| round 150 | 365 | 3.0% | 1,663 | 8.0% |

Rarely on offer (0.5% and 2.5% of sends), and self-play has not taken them
up ("surplus" creeps up, "hold" does not). They cannot matter much either way.

**A bigger finding on the way: self-play has made the amounts nearly random.**

| network | amount uncertainty at its sends (share of a uniform pick) | favourite amount gets | route it took had |
|---|---|---|---|
| imitation start (copy of strategic_ai) | 13% (median 4%) | 89% | 65% |
| deep_from_imitation round 500 | 88% (median 96%) | 27% | 26% |
| round 900 (ab_plain round 400) | 86% (median 95%) | 29% | 26% |
| large network, round 150 | 82% (median 92%) | 33% | 28% |

The most used amount covers only 15% of the sends (imitation start: 27%),
and the self-play networks send about twice as often (69,000 sends in 48
games against 34,000). They became stronger anyway, presumably through
which route and when; how much is sent has become close to a coin toss.
Likely cause: the entropy bonus (weight 0.01) pushes every choice towards
even in every position, while the signal for "how much" (one advantage for
the whole move, from noisy game results) is too weak to resist it over
hundreds of rounds. Not yet tested.
- [x] How much always playing the favourite gains (round 900, 100 fresh
      maps, both seats; `ml/runs/logs/round900_sampling.txt`, `..._greedy.txt`):

| round 900 against | sampling | always the favourite |
|---|---|---|
| aggressive_ai | 94.8% (188–9) | 97.5% (193–3) |
| strategic_ai | 85.5% (169–27) | 88.8% (173–18) |
| overall | 90.1%, median game 54 s | 93.1%, median game 36 s |

  A little better (+3 points, about one standard error) and much faster
  wins, although its favourite amount has only ~29% probability: the
  randomness in the choices costs something, but the scripted AIs leave
  little room to show how much.
- [x] Test: continue round 900 with a much smaller entropy bonus (0.001),
      or with the bonus on the route only, against plain continuation.
      → part of the shaping experiment below.

## Rules of thumb as potentials: production, the front, and a smaller entropy bonus (2026-10-04)

**Where the sends go** (48 self-play games, fresh maps; `ml behaviour`, new):

| | sends per game and seat | to own bases | units away from the front |
|---|---|---|---|
| imitation start | about 320–485 | 58–60% | 40% |
| round 900 | 650–810 | 61–77% (69% on maps 990,000+) | 31–32% |

The share sent to own bases depends a lot on the maps (61% on 12 maps, 69%
and 77% on two sets of 48); the clear difference to the imitation start is
how often round 900 sends (about twice as often), and how random its
amounts are (above). Only about 10% of its own-base sends leave a base that
enemy fleets are heading for, so most of it is shuffling between its own
bases, not retreating under pressure. Units in the rear (bases whose
neighbours are all mine, or on their way there): 18% for both on average,
but 30% after the first minute for round 900 against 23%.

**The user's ideas, and the form chosen:** reward controlling more of the
production, defending, capturing; punish units that are not at the front.
As event bonuses these can be gamed (lose a base and take it back for a
second capture bonus; leave a neutral pocket inside your territory so the
bases around it count as front). So both are built as **potentials**:
every move is credited with how much it changed a measure of the position,
and giving something back costs exactly what gaining it earned. In theory
(Ng, Harada & Russell 1999) this never changes which strategy is best,
only how soon a move gets credit; in practice that is the point, as our
credit for a capture now only arrives through noisy game results. The
existing "material hint" was already a potential, but it fades out after 50
rounds and has been off since.

- `--production-potential W`: W x (2 x my production share - 1). Captures,
  defences and losses all count, at once.
- `--front-potential W`: minus how far my units are from the enemy, on
  average: each unit at one of my bases or on its way to any base counts
  with the shortest path from there to the nearest enemy base, as a share
  of the map's longest shortest path. Distance instead of "all neighbours
  mine", so a neutral pocket is not a loophole and the early game (no
  enemy next door) is not distorted. Computed from what is recorded
  anyway: the networks' inputs do not change.
- Every round now logs `behaviour/...`: sends to own bases, sends per game
  and seat, units away from the front.

**Experiment** (`ml/scripts/done/shaping_arms.sh`, started 2026-10-04 00:55):
four arms continue round 900 for 150 rounds with the control's settings
(`small_selfplay`, run on 2026-10-03, is the plain control):
entropy 0.001; production potential 0.5; front potential 0.5; all three.
Then `ml compare` of all five runs and `ml behaviour` of their last rounds.
- [x] `shaping_entropy` (00:55–02:35): in-run checks against round 900
      (16 games each, rough) mostly 69–88% in the second half (plain
      control: mostly 40–60%). Entropy 3.35 → 2.4; sends to own bases
      78% → 84% (more, not fewer); units away from the front unchanged (33%).
- [x] `shaping_production` (02:35–04:26): in-run checks against round 900
      mostly 50–63%, ending at 38%: no visible effect. Behaviour unchanged
      (sends to own bases 76–81%, units away from the front 32–38%);
      entropy stays around 3.
- [x] `shaping_front` (04:26–06:17): in-run checks against round 900
      50–75%, against strategic_ai often 94–100% in the second half. But
      units away from the front did not fall (33% → 34–40%), so at weight
      0.5 the potential did not produce the behaviour it rewards.
- [x] `shaping_all` (06:17–08:04): entropy 3.0 → 1.6, sends per game and
      seat 740 → 520; in-run checks against round 900 56–78%.
- [x] Comparison (`ml/runs/compare_shaping/`, 100 games per match):

| round | plain | entropy 0.001 | production | front | all three |
|---|---|---|---|---|---|
| 0 (= round 900) | 1310 | 1333 | 1325 | 1308 | 1308 |
| 50 | 1302 | 1362 | 1307 | 1337 | 1357 |
| 100 | 1290 | 1397 | 1325 | 1328 | 1343 |
| 150 | 1318 | **1416** | 1307 | 1322 | 1390 |

  Round 0 is the same network five times: 1308–1333 is the measuring
  noise. Last rounds head to head: entropy 0.001 beats plain 67–33% (+3.4
  standard errors), production and front 66% each (+3.2); against all
  three 50.5%. Production and front against plain 54.5% and 52% (within
  the noise); all three beats plain 60% (+2.0).

- [x] Behaviour of the last rounds (`ml/runs/logs/shaping_behaviour.txt`, 48 games each):

| | round 900 | plain | entropy | production | front | all three |
|---|---|---|---|---|---|---|
| sends per game and seat | 659 | 645 | 699 | 740 | 621 | 643 |
| sends to own bases | 69% | 71% | 75% | 73% | 67% | 74% |
| units away from the front | 32% | 33% | 33% | 34% | 35% | 34% |
| amount uncertainty (100%: even) | 85% | 84% | 69% | 78% | 74% | 58% |
| favourite amount's probability | 30% | 30% | 44% | 36% | 39% | 53% |

**Conclusions**
- **The entropy bonus was holding self-play back.** At 0.001 instead of
  0.01, self-play from round 900 improves again: +80–100 rating in 150
  rounds, still rising at the end (1397 → 1416), where plain stayed flat.
  The amounts become more decided (uncertainty 85% → 69%), though still far
  from the imitation start (12%).
- **The two potentials add nothing measurable**, alone or on top of the
  entropy change, and **neither changed the behaviour they reward**: sends
  to own bases and units away from the front are the same in every arm.
  At weight 0.5, their credit is small next to the rest of the advantage;
  shuffling is either not costly enough to show in results, or needs a
  much stronger signal than these potentials give. Not pursued further for now.
- `shaping_entropy` round 150 is the strongest network on instinct; on
  instinct against deep_from_imitation round 500: 68.5% (62–25, 13 draws).
- [x] Look-ahead check (`ml/runs/shaping_entropy/check.json`, judge2):

| test | result |
|---|---|
| on instinct vs deep_from_imitation round 500 (100 games) | 68.5% (62–25, 13 draws) |
| look-ahead vs Beast (100 games) | 52.5% (48–43, 9 draws) |
| look-ahead vs Atlas (60 games) | 80.8% (47–10, 3 draws); Beast had 82.5% |

  Not clearly stronger with the look-ahead, so **Beast stays**. The same
  pattern as ab_plain round 150: a better network on instinct does not
  make a better look-ahead player, where the judge decides between its
  candidates. To improve Beast, the judge itself would have to improve,
  e.g. a judge trained on games of the stronger networks (step 1 of the
  judge loop again).
- [x] Next: continue `shaping_entropy` (still rising at round 150) → below.

## The front runner's whole road, and a shrinking entropy bonus (2026-10-04)

**`ml lineage CHECKPOINT`** (new, `ml/evaluation/lineage.py`) follows a
checkpoint's history and joins the runs that trained it into one run
folder, numbered from the first self-play round on, for `ml monitor --run
lineage_NAME`. It adds each run's entropy bonus as a chart, renames "vs
starting model" to "vs the start of its run" (a different model per run),
and adds the ratings from the comparisons that rated each run, chain-linked
(each run's ratings shifted so its round 0 equals the previous run's last
rated round, the same network). Comparisons now record which runs they
rated, so later ones are found by themselves.

For the front runner (`ml monitor --run lineage_shaping_entropy`):

| rounds | run | hours | entropy bonus | rating (linked) |
|---|---|---|---|---|
| (imitation of strategic_ai, 20 epochs) | deep_imitation | | | |
| 0–500 | deep_from_imitation | 5 h 35 | 0.01 | 969 → 1357 |
| 500–900 | ab_plain | 4 h 48 | 0.01 | 1357 → 1405 |
| 900–1050 | shaping_entropy | 1 h 46 | 0.001 | 1405 → 1487 |

The entropy chart shows the story: under the 0.01 bonus the policy went
from very decided (0.4 at the imitation start) to nearly undecided (about
3.4) within 200 rounds and stayed there while progress slowed and stopped;
after the switch to 0.001 at round 900 it falls (2.4 at round 1050) and the
rating climbs again. (Linked ratings carry the noise of three comparisons,
about ±15 each.)

**Shrinking entropy bonus** (`--entropy-half-life N --entropy-floor F`):
the bonus halves every N rounds of the run, from `--entropy-weight` towards
the floor, never below, so the network keeps trying things. Logged every
round as `entropy bonus/weight`. For a fresh run this would be e.g. 0.01
halving towards 0.0005; the front runner is already at 0.001.

**Run** (`ml/scripts/done/entropy_decay.sh`, started 2026-10-04 09:50): the front
runner continues for 300 rounds (seed 1, new maps), bonus 0.001 halving
every 200 rounds towards 0.0002 (0.0006 at round 200, 0.0004 at 300); then
`ml compare` of rounds 0/100/200/300 and the lineage view updated.
- [x] Results (crashed once at round 190 and resumed, see "Colab, and a crash
      at round 190"; finished 14:20; `ml/runs/compare_entropy_decay/`):

| entropy_decay round (lineage round) | 0 (1050) | 100 (1150) | 200 (1250) | 300 (1350) |
|---|---|---|---|---|
| rating | 1404 | 1410 | 1408 | 1394 |
| vs strategic_ai | 92% | 92% | 92% | 90% |

  **Flat.** Lowering the bonus from 0.01 to 0.001 was a one-off step (about
  +100 as the policy sharpened); shrinking it further, to about 0.0004, gave
  nothing more in 300 rounds. The 64×6 network has plateaued again, at about
  1480 on the linked scale (`ml monitor --run lineage_entropy_decay`). The
  best network on instinct stays `shaping_entropy` round 150 (or any of
  these rounds: they cannot be told apart).
- [x] Checked more closely (2026-10-04): round 300 against round 0 head to
      head, 200 games on fresh maps: **54.2%** (95–78, 27 draws), about 1.2
      standard errors above even, so at most a small gain (the in-run checks'
      64–68% against the start in rounds 200–300 were the top of the noise).
      No learning problems after the crash guard went in. Behaviour (48 games
      each, `ml/runs/logs/entropy_decay_behaviour.txt`):

| round | 0 | 100 | 200 | 300 |
|---|---|---|---|---|
| sends per game and seat | 721 | 639 | 609 | 582 |
| sends to own bases | 77% | 79% | 76% | 74% |
| units away from the front | 33% | 33% | 33% | 32% |
| amount uncertainty (100%: even) | 69% | 65% | 56% | 52% |
| favourite amount's probability | 45% | 49% | 55% | 59% |

  The smaller bonus keeps making the network more decided (amounts 69% →
  52% of an even pick) and it sends less (−19%), but that no longer turns
  into strength, and shuffling between its own bases and units far from
  the front stay where they were.

## Inputs (from 2026-10-02)

**Question:** is the network missing what it needs to see? The capacity
probe says copying `strategic_ai`'s sends stalls at 67–72% whatever the
size, while eighths could reproduce 93%. `strategic_ai` decides from
quantities the inputs make hard to work out:

| `strategic_ai` works out | What the network had |
|---|---|
| Strength at a target when a fleet would land: garrison, plus production until then, plus or minus only the fleets landing by then | Units on a log scale, incoming fleets as one total whenever they land, a "future" at a different moment per base, and only a yes/no for "just enough" |
| Spare units at a source: units minus a reserve that depends on bordering a base it does not own and on incoming fleets | Pieces of it, on a log scale |
| Counts and totals over neighbours: number of routes, neighbours not its own, spare units around a target | Nothing direct; attention averages, so "how many" and "how much in all" are awkward for it |
| Shortest path through its own territory only | Fixed shortest paths through any base |

The log scale matters most: fights are subtraction, and 41 against 45 units
is about 0.02 apart on it.

**Built (step 1 of 4):**
- `ROUTE_FEATURES`, a new table recalculated every tick: for every route,
  in each direction, from the point of view of whoever owns the source,
  the units a fleet sent now needs to make the target theirs when it lands
  (`capture_cost`, log scale), that as a share of the source's units
  (`capture_cost_share`; above 1 is out of reach), and the seconds of the
  target's production that earn it back (`payback_time`). Routes into one's
  own threatened bases give the cost of holding them; routes from enemy
  bases into ours are the threats. The calculation is the one "just enough"
  already used, moved to `forecast.units_to_capture`.
- Production shares, your idea: `my_production_of_all`,
  `strongest_enemy_production_of_all`, `second_enemy_production_of_all`,
  all out of all production on the map, neutral included
  (`neutral_production_share` was already that). Enemies are ranked by
  production so the slots mean the same in every match, and a missing or
  eliminated enemy is 0: a 1v1 is then a three-player game whose third
  player has nothing, with no special "absent" input. In a 1v1 these follow
  exactly from `my_production_share` and `neutral_production_share`, so they
  are mainly the three-player-ready form. Full three-player support also
  needs the base features to tell the two enemies apart.
- **Old models keep working.** Features are only ever added at the end of a
  list. A model reads the first N of each, as many as it was trained with
  (`base_features`, `pair_features`, `route_features`, `global_features` in
  its settings), and loading accepts a checkpoint whose lists are the start
  of the current ones. Checked: `deep_from_imitation` round 500 gives
  exactly the same outputs as under the old code (Atlas's copy of it) on 192
  positions. Atlas's copy of the encoders is unchanged, so Atlas can only
  forecast with networks that do not use the new features.
- `ml show-state` prints the route features of one base, next to the units
  needed worked out again.
- Recorded data per decision grows by the route values (about 0.7 KB on a
  median map).

**The check** (`ml/scripts/done/input_check.sh`): the capacity probe's "deeper"
run again (64×6, the same 600 teacher games, 20 epochs), with three input
sets, each on the teacher games of seed 0 and seed 1: as before, plus
routes, plus routes and production. "Before" on seed 0 should repeat
`capacity_deeper`. Agreement on sends is the measure: it costs about 12
minutes a run and involves no opponent's luck.

**Result (2026-10-02)**, `ml/runs/logs/input_check_report.txt`, 64×6, 20 epochs:

| Inputs | Sends copied, seed 0 / seed 1 | Check loss, seed 0 / seed 1 |
|---|---|---|
| before | 73.5% / 66.2% | 0.348 / 0.438 |
| + routes | 72.4% / 67.7% | 0.374 / 0.422 |
| + routes + production | 69.9% / 63.3% | 0.377 / 0.456 |

- **The route features did not help.** On each seed they differ from
  "before" by about 1 point, one way on one seed and the other way on the
  other.
- **The noise, measured:** "before" on seed 0 is `capacity_deeper` again,
  with the same inputs, seed, and starting weights, and came out 73.5%
  against 70.7%: training on the Apple GPU is not exactly repeatable. Seed
  1's teacher games are harder to copy (every input set is 5–7 points lower
  there). Compare within a seed, and treat about 3 points as noise.
- **What this rules out:** the network had the capture cost of every route,
  worked out exactly and as a plain share of the source's units, and still
  copies about 30% of the sends wrongly. Working out "how much does this
  capture need" was not what it lacked.
- **For the judge** (see "The judge"): no input set is better, so the inputs
  as before are the safe choice. New networks read the new inputs by
  default (they are at the end of the lists); `--set route_features=0
  global_features=4` leaves them out.
- **Next:** look at the mistakes instead of guessing at inputs (below).

**The mistakes, taken apart (2026-10-02)** (`ml mistakes`, report in
`ml/runs/logs/mistakes_report.txt`): `strategic_ai` on 120 fresh maps against
itself and the two simpler AIs, 117,432 decisions, 48,305 of them sends;
the network is `inputs_before_seed0` (the routes network looks the same).
Each send by the teacher, sorted by which part of `strategic_ai` made it
and by what the network's most likely move was instead:

| Network's move | Defence (3% of sends) | Attack (20%) | Pressure (22%) | Reinforcement (54%) | All sends |
|---|---|---|---|---|---|
| the same move | 13% | 69% | 75% | 73% | 70.6% |
| same route, same units under another option | 1% | 1% | 2% | 1% | 0.9% |
| same route, neighbouring amount | 10% | 9% | 5% | 4% | 5.4% |
| same route, amount further off | 6% | 2% | 1% | 0% | 0.8% |
| **a route the teacher uses itself within the next second** | **39%** | **12%** | **11%** | **11%** | **12.0%** |
| another route | 9% | 2% | 4% | 10% | 6.7% |
| waits | 23% | 6% | 4% | 2% | 3.7% |

- **"Sends copied" undercounts.** `strategic_ai` acts on one plan per
  tick and works through the others on the next ticks. In 12% of its sends
  the network picks one of those next plans first: the teacher's own move,
  in another order. Another 6% are the same route with the next amount
  option up or down. Counting those, the network plays about 89% of the
  teacher's sends; only about 11% are clearly different moves.
- **Reinforcement was not the weak part** (my guess): 73% exact, with most
  of the rest "next plan first".
- **Defence is the real gap:** only 13% exact, and in 23% of the moments
  `strategic_ai` reinforces a base under attack, the network does nothing.
  Defence is 3% of sends, so about 1% of all decisions: few examples to
  learn from, and the routes table (which shows the cost of holding a
  threatened base) did not change it.
- **What this says about the inputs:** the network sees enough to play
  almost all of `strategic_ai`'s moves. The inputs are not the main limit
  for copying it. The 67–72% that started this was mostly a strict
  measure. (Copying `strategic_ai` only tests what `strategic_ai` uses;
  stronger play could still need more.)

**Remaining steps, on hold until the mistakes are understood:** units on a straight scale as well
(step 2), neighbourhood totals (step 3), distance to the nearest border
through one's own territory (step 4), each measured the same way; then
self-play with the best set.

## Losses to the weak AIs (2026-10-02)

**Tools:** `ml evaluate` saves every game the candidate loses to
`aggressive_ai` (or to the AIs named with `--save-losses-against`) to
`ml/runs/losses/<candidate>/`: the map, the seats, the seeds, and every move.
`ml rematch FILE` plays that start again, 50 times from each seat, and counts;
`ml watch FILE` shows the recorded game in the game window (`--live`: the
players play it again).

**Measured** (`ml/scripts/done/weak_ai_losses.sh`, log `ml/runs/logs/weak_ai_losses_log.txt`):
`deep_from_imitation` round 500 against `aggressive_ai` and `careful_ai`, 200
games each on 100 fresh maps.

| Way of playing | Losses to `aggressive_ai` | Losses to `careful_ai` |
|---|---|---|
| picking moves by chance (as in training) | 6 | 9 |
| always its top move (`--greedy`) | 4 | 8 |
| move scoring, 8 candidates | 4 | 6 |

- **The same few maps keep coming back**, and nearly always from one seat:
  the rematches are 50–0 for the opponent from the losing seat and 50–0 for
  us from the other.
- **Some of these maps are decided by the seat.** Checked by letting other
  players take the losing seat: on `900075` (7 bases) and `900094` (11
  bases), `strategic_ai`, `careful_ai`, and our network all lose to
  `aggressive_ai` from red. No player could win there; the map generator's
  asymmetry made that seat lost.
- **Others are real weaknesses**, because `strategic_ai` wins where we lose:
  `900064` from red against `aggressive_ai`, `900118` from blue against
  `aggressive_ai`, and `900128` (4 bases) against `careful_ai` from both
  seats (where `strategic_ai` draws).
- **Picking by chance adds a few avoidable losses**: on some maps it loses
  10–40% of rematches from a seat it usually wins from.
- **Consequence for measuring:** "never loses to `aggressive_ai`" can only
  be asked of maps where the seat does not decide. A fair yardstick is
  whether we lose where `strategic_ai` (or Atlas) wins from the same seat.

**One lost game traced** (saved game 30: move scoring on
`deep_from_imitation` round 500, red, against `aggressive_ai` on `900064`,
27 bases; `strategic_ai` wins from red here). Red held 1 base for 25 seconds
while blue took 7; 228 of red's 302 sends went to its own bases.

- **The value output knew it was losing from 5 seconds on** (−0.46, then
  −0.5 to −0.9).
- **The policy wanted to expand; move scoring overruled it.** At 10 s the
  policy gave capturing the empty neighbour `briar` 77%, but the value
  output rated the position after it at −0.60 against −0.52 for waiting, so
  move scoring waited (also at 15 and 20 s). At 30 and 35 s the policy
  preferred capturing `juniper` (59%, 63%); the value output preferred moving
  1 unit to its own base (−0.535 against −0.575), and that was played.
- **So here the weak part is the value output's judgement of small
  differences**, not what the policy learned: gaps of 0.04–0.08 between
  moves decide, and the value output is trained only to predict the final
  result, which it does roughly (error about 0.08 squared, against 0.3).
- What this suggests for training: a more precise value output (targets
  that look a few seconds ahead instead of only at the end, or its own
  network), and possibly a permanent production-share term added in the
  potential-based way that cannot change which play is best.

**The game window now shows the players' names** (`renderer.py`, `main.py`;
also in `ml watch`): in the header instead of RED/BLUE, long names on two
lines, and the winner's name under the result banner. The terminal line at
the end names the winner too.

## Network size (from 2026-10-01)

**Question:** is the network (120,120 weights) too small, and if so, should
it be wider, deeper, or something else?

- **What width and depth mean here.** Attention is all-to-all, so every
  base already sees every other base in the first layer; depth does not
  widen the view. Instead it adds reasoning steps that build on each other
  ("X's defenders are away, so X is weak, and Z gets there first"). Width is
  how much each base's description can hold.
- **A third candidate: the route head.** Every send decision (where, and
  how much) is scored by a single hidden layer from the source, the target,
  and the pair. New setting `route_layers` (default 1, the network as it
  was; old checkpoints load unchanged) adds more layers there.
- **Evidence so far.** Copying `strategic_ai`, which never uses chance, the
  network matches 99% of its waits but only 64–67% of its sends, with the
  training loss still falling and a small gap to the checking set (0.35 and
  0.42). It is somewhat short of capacity, and the shortfall is in the sends.

**Cost of larger networks** (measured on real positions; GPU timings are rough):

| Width × layers (heads) | Weights | Playing, µs per decision (one CPU thread) | Learning, ms per batch of 1024 (GPU) |
|---|---|---|---|
| 64×3 (4), current | 120k | 70 | 85 |
| 128×3 (4) | 470k | 115 | 101 |
| 64×6 (4) | 221k | 112 | 104 |
| 128×4 (8) | 603k | 166 | 147 |
| 128×6 (8) | 869k | 224 | 191 |
| 192×4 (8) | 1.35M | 260 | 216 |

The network is a small part of a self-play round's playing time, so 128×4
should make a round about 20–40% slower, not twice as slow. Search slows
down by the playing factor. A larger network cannot inherit round 500's
weights, so the long run would start over.

**The probe** (`ml/scripts/done/capacity_probe.sh`): six
variants imitate `strategic_ai` on the same 600 teacher games for 20 epochs
with the same settings: current, wider (128), deeper (6 layers), route head
(2 route layers), bigger (128 wide, 4 layers, 8 heads, 2 route layers;
620k weights), and largest (192 wide, otherwise as bigger; 1.38M weights).
The largest variant is the upper bound: if even it does not copy the sends
better, size is ruled out. Report: `ml/runs/logs/capacity_probe_report.txt`, made
with `ml summary`. About 1 h 20 min.

- If a variant clearly beats 64–67% agreement on sends, that is where
  capacity is missing. That variant then gets about 150 rounds of self-play
  from scratch, compared with the overnight run at the same round.
- If none does, the limit is in the inputs or the data, and a larger network
  will not help self-play either.
- Caveat: all variants use the imitation trainer's step size (1e-3 falling to
  1e-4), which may not suit every size equally.

**Probe result (2026-10-01)**, one seed each, same 600 teacher games, 20
epochs; full table in `ml/runs/logs/capacity_probe_report.txt`:

| Variant | Weights | Time | Same move as teacher on its sends | Move loss, training | Move loss, check | Value loss, check |
|---|---|---|---|---|---|---|
| current (64×3) | 120k | 7 min | 67% | 0.351 | 0.411 | 0.162 |
| wider (128×3) | 470k | 12 min | 72% | 0.332 | 0.392 | 0.164 |
| **deeper (64×6)** | 221k | 11 min | 71% | 0.316 | **0.360** | 0.151 |
| route head (64×3, 2 route layers) | 124k | 9 min | 71% | 0.363 | 0.416 | 0.148 |
| bigger (128×4, 8 heads, 2 route layers) | 620k | 13 min | 71% | 0.297 | 0.368 | 0.183 |
| largest (192×4, 8 heads, 2 route layers) | 1.38M | 17 min | 70% | 0.323 | 0.422 | 0.184 |

- **Size is not the main limit.** Every larger variant copies the sends a
  little better (70–72% against 67%), but none comes close to copying a
  teacher that never uses chance and keeps no memory between ticks. The
  amount options alone cap this at about 93% (eighths reproduce 93% of
  `strategic_ai`'s sends exactly), so most of the remaining gap is in what
  the network is shown or in the amount of data, not in its size.
- **Depth beats width.** Six layers at width 64 (221k weights) checks best
  (0.360), better than twice the width (0.392 with twice its weights). A
  deeper route head alone changes nothing (0.416).
- **The largest networks overfit 540 training games.** The gap between
  training and check loss grows with size (0.06 for current, 0.10 for
  largest), and their value check is the worst (0.18). Self-play has far
  more data (48,000 games overnight), so this is a smaller concern there.
- **Caveats:** one seed each, and the check loss of a single run moves by
  a few hundredths from epoch to epoch (route head: 0.536 at epoch 10, 0.540
  at 15). Differences under about 0.03 are not meaningful. Every curve was
  still falling at epoch 20, cut off by the step-size schedule.
- **Candidate for the long run:** deeper, 64×6. Playing costs 1.6× the
  current network (112 µs against 70 µs per decision), far less than the
  wide variants.
- [x] `route_layers` setting, `ml checkpoint new --set`, `ml summary`, with tests
- [x] Same setting copied into Atlas's copy of the network (`atlas_model/models/graph_transformer.py`), so Atlas still loads our new checkpoints
- [x] Run the probe
- [ ] Self-play check of the winner: part of the long run (see "Next steps")

## Move scoring: a value-only player (from 2026-10-01)

**Idea (yours):** a model that predicts who wins from a position, and a
player that simply picks the move whose result it likes best, as chess
engines pick moves by their evaluation.

**Built** (`ml/play/value_player.py`, player name `value:PATH`): for every
legal move, the position right after it (units leave the source, a fleet is
on its way), all judged in one batch by the network's value output; the
best is played. No simulation: the encoder already works out what fleets in
flight will do, so a send shows up in the position the moment it is made.
This is TD-Gammon's way of playing. `?candidates=N` judges only the policy
output's N favourite sends (plus waiting).

- **Cost** (overnight round 500, one CPU thread): 2.2 ms per decision at the
  median on 8 bases and 5.4 ms on 24 bases, judging every legal move; 1.5
  and 2.3 ms with 8 candidates. As fast as the plain network, about 10 times
  cheaper than search.
- **Blind spots:** it does not look at the opponent's reply (the value output
  has to have learned it), and it will find the positions the value output
  overrates, since it always picks the highest.
- **Shared code:** `ml/play/players.py` turns `search:` and `value:` names into
  players for the evaluator and the rollouts, with settings after `?`.
- [x] Player, names, tests (`ml.tests.test_value_player`)
- [x] Strength (`ml/scripts/done/value_player_check.sh`, log in
      `ml/runs/logs/value_player_check_log.txt`): both versions against the plain
      network, `strategic_ai`, and search, all on round 500, 80 games per
      pairing on 40 fresh maps

**Result (2026-10-01):**

| Player (all round 500) | vs plain network | vs search | vs `strategic_ai` | Overall |
|---|---|---|---|---|
| move scoring, every legal move | 27.5% (21–57, 2 draws) | 21.2% (17–63) | 51.9% (40–37, 3 draws) | 33.5% |
| **move scoring, the policy's 8 favourite sends** | **64.4% (50–27, 3 draws)** | **58.8% (45–31, 4 draws)** | **83.8% (67–13)** | **69.0%** |
| for reference: plain network vs `strategic_ai` | | | 69–74% in earlier checks | |

- **Judging every move makes it much worse.** Among about a hundred
  resulting positions, the highest value is mostly one the value output
  overrates. Picking the maximum of a noisy judge finds its mistakes.
- **With the policy as a filter it is the strongest player we have.** It
  beat the plain network 50–27 (about 2.7 standard errors from even), beat
  search 45–31 at a tenth of its cost (2 ms per decision against 30), and
  scored 84% against `strategic_ai`. The policy proposes sensible moves; the
  value output picks among a few; neither alone is as good.
- **Caveats:** one network, 80 games per pairing; the number of candidates
  (8) was a guess, not tuned.
- **What it opens up:** a cheap teacher. Distillation with search as the
  teacher costs about a minute per game; with this player it would cost
  about what plain self-play does. The same goes for using it inside
  self-play to generate better games.

**Learning from it** (`ml/scripts/done/value_distill_test.sh`, started
2026-10-01 20:25): the distillation loop now takes `--teacher value`
(`--value-candidates 8`). From round 500, 10 iterations of 150 teacher games
each; then the plain network at iterations 0, 5, 10 rated on fresh maps, and
move scoring on the final network against move scoring on round 500.
About 1.5 hours.

**Result (2026-10-01): it made the network worse, steadily.** Log in
`ml/runs/logs/value_distill_test_log.txt`, ratings in
`ml/runs/compare_value_distill/report.txt`.

| Iteration | Rating (`strategic_ai` = 1000) | vs `strategic_ai` | Quick check vs round 500 (64 games) |
|---|---|---|---|
| 0 (round 500) | 1195 | 72% | |
| 1 | | | 49% |
| 3 | | | 41% |
| 5 | 1069 | 57% | 24% |
| 10 | 968 | 52% | 20% |

Move scoring on the final network lost to move scoring on round 500 by
17–60 (3 draws), and scored 35% against the plain round 500: the teacher
got worse along with its pupil.

- **What changed in the network** (same fixed positions from `strategic_ai`
  games): the chance of waiting went from 4% to 28% after one iteration
  (11% by iteration 5), and the policy's entropy from 3.8 to 2.5. The value
  output's error on those positions did not get worse.
- **Why, most likely:** move scoring always plays its single best-judged
  move, waiting whenever no send judges better. Copying that pulls the
  network away from its own way of playing (sending a little on almost
  every tick) towards a narrower, deterministic one, without the judgment
  that made the teacher's choices good. The teacher is the network itself,
  so each iteration's teacher inherits the damage, and its value output
  learns from ever narrower games (both seats the same deterministic
  player). The network never caught up with its teacher: same move 61%
  after the first iteration, about 30–40% after that.
- **Flaw found on the way:** the games remembered from the previous
  iteration are lost when a run resumes in a new process (iteration 6, the
  first after the restart, learned from 77,000 decisions instead of
  155,000). It does not explain the decline, which began in the first
  process.
- **Decision:** move scoring is a way to *play*, not to *teach*, for now.
  Train with plain self-play and play with `value:PATH?candidates=8`.
  Learning from it would need softer targets (a spread over the candidates
  weighted by their values, not only the best one) and a pull back towards
  the network's own policy; that is AlphaZero's recipe, and an experiment
  of its own.

## Live feedback during training (requirement for Stages 7–8)

- **Terminal:** one line per training round: games played, results against
  the scripted AIs, and the learning numbers.
- **Log file:** every round appended to `metrics.jsonl` in the run folder.
- **Live window:** `python -m ml monitor --run NAME`, a separate program that
  reads the log and draws the curves, so it can be opened and closed without
  touching training.
- **Watch it play:** trainers keep `ml/runs/latest.pt` up to date, and
  `neural_ai` re-reads it at every new match (already built).
- **Checkpoints:** every round is saved (0.5 MB each, weights only), plus
  `latest.pt` with the optimizer state for resuming. 1,000 rounds is about
  0.5 GB. Finished runs can be thinned out by hand (as in the housekeeping),
  but a league run needs its champions' round files until it is done.

## Measurements

| Date | What | Result |
|---|---|---|
| 2026-09-30 | `Game.step` | 1–7 µs per tick |
| 2026-09-30 | `Game.view_for` | 30–180 µs per call (grows with map size) |
| 2026-09-30 | Full scripted match, one core | 0.01–0.5 s; 250–900 ticks when decided |
| 2026-09-30 | Share of ticks on which scripted AIs send | 6–74% |
| 2026-09-30 | Stand-in network (3 layers, d=64, 32 bases, 100k weights), one position, CPU | 0.25 ms |
| 2026-09-30 | Same, batch 256, CPU 8 threads | ~11k decisions/s |
| 2026-09-30 | Same, batch 256, Apple GPU (MPS) | ~120k decisions/s |
| 2026-09-30 | Bundled maps with unequal starting positions | 7 of 11 (`crossroads`, `four_corners`, `frontier`, `siege`, `three_fronts`, `triangle`, `triple_ring`) |
| 2026-09-30 | `strategic_ai` vs itself, no noise, run twice | Identical result both times on all 11 maps: one fixed outcome per map, so no win *rate* exists |
| 2026-09-30 | Same on `ring` (a mirrored map), 10% of moves dropped, 60 games | Red wins 100% with the original base IDs, 40% with IDs re-randomised: the 100% was alphabetical tie-breaking, not the map |
| 2026-09-30 | Same probe, maps where one seat wins 95%+ | `crossroads` (Red), `four_corners` (Blue), `frontier` (Blue), `siege` (Blue), `triple_ring` (Red) |
| 2026-09-30 | Same probe, other maps (seat win rates in order Red/Blue/Green) | `salt_and_pepper` 45/55, `ring` 40/60, `nomads` 62/38, `triangle` 28/67/5, `three_fronts` 32/2/67, `classic` 83% draws |
| 2026-09-30 | Map generation speed | 2.1 ms per map |
| 2026-09-30 | Generated maps, 2000 seeds: bases per map | 4 to 29, median 17 |
| 2026-09-30 | Same: travel time | median 2.5 s; 80% between 1.5 and 4.3 s |
| 2026-09-30 | Same: production | median 1.2; 80% between 0.5 and 3.05; never exactly 0; reaches the limit of 10 rarely |
| 2026-09-30 | Same: neutral units | median 15; 80% between 6 and 37; largest seen 211 |
| 2026-09-30 | Same: starting units | median 25; 80% between 10 and 63; reaches the limit of 250 rarely |
| 2026-09-30 | `strategic_ai` vs `aggressive_ai` / `careful_ai`, 300 generated maps, both seats | 93–94% decided within 180 s (bundled 1v1 maps: 100% of 8); median length 64–72 s (bundled: 54–73 s) |
| 2026-09-30 | `aggressive_ai` vs `careful_ai`, same maps | 66–68% decided (bundled: 7–8 of 8) |
| 2026-09-30 | Probe on 240 mirrored generated maps (10% of moves dropped, 10 games each) | Red 33.9%, Blue 36.6%, draws 29.5%; difference -2.7 ± 1.9 points, so no seat bias |
| 2026-09-30 | Probe vs asymmetry level, 60 maps x 20 games: maps where one seat wins 90%+ | 0.02: 25/60 · 0.05: 29/60 · 0.1: 38/60 · 0.2: 42/60 · 0.3: 49/60 · 1.0: 59/60 |
| 2026-09-30 | State encoding, one position | 30–45 µs (6 to 27 bases); once-per-match layout 20–80 µs |
| 2026-09-30 | Stacking 256 encoded positions into a batch | 0.5 ms |
| 2026-09-30 | Bases with a fleet on its way, in `strategic_ai` vs `aggressive_ai` matches | 26% of the time; of those, 24% will change owner |
| 2026-09-30 | Amount options vs `strategic_ai`'s 6,882 sends: share reproduced exactly / mean unit error | halves 40% / 33% · quarters 54% / 17% · eighths 93% / 1.6% · tenths 96% / 0.8% |
| 2026-09-30 | How often "just enough" is the closest match to a scripted AI's send | 1–10%; the scripted AIs add a buffer or send a fixed share |
| 2026-09-30 | Scripted AI moves rejected by the game, 40 maps x 4 pairings | 0 |
| 2026-09-30 | Choices on offer per position (eighths) | median 14 routes and 114 sends; largest seen 90 routes and 720 sends |
| 2026-09-30 | Cost per decision before the network: `view_for` + state encoding + legal choices | 42 + 47 + 69 = about 160 µs, so roughly 6,000 decisions/s per core |
| 2026-09-30 | `graph_transformer` size | 120,120 weights, 100,416 of them in the three attention layers |
| 2026-09-30 | `graph_transformer` speed, one position, CPU | 0.8 ms |
| 2026-09-30 | Same, batch of 256 | CPU about 7,000 decisions/s; Apple GPU about 24,000–28,000 decisions/s |
| 2026-09-30 | Where the GPU time goes (batch 256, 29 bases) | attention layers 6.6 ms, scoring every pair of bases about 4 ms |
| 2026-09-30 | Checkpoint file size (`graph_transformer`, default settings) | 0.5 MB |
| 2026-09-30 | One decision by the agent (encode + network + decode), CPU | 0.7 ms on an 8-base map, 1.2 ms on a 26-base map |
| 2026-09-30 | Full evaluation, 228 games on 10 processes | about 30–60 s |
| 2026-09-30 | Rollout speed, 6 processes: untrained network against itself | 8,300 decisions/s, 3.5 games/s (most games run the full 180 s) |
| 2026-09-30 | Same: untrained network against `strategic_ai` | 11,700 decisions/s, 10 games/s |
| 2026-09-30 | Same: `strategic_ai` against itself, both seats recorded (imitation data) | 21,000 decisions/s, 8 games/s |
| 2026-09-30 | Scaling with processes (network against itself) | 1: 2,900/s · 2: 5,500 · 4: 8,800 · 6: 10,600 · 8: 10,200 · 10: 11,400. Flat after about six; cause not established |
| 2026-09-30 | Running each worker's model on the GPU instead of the CPU | Slower at every worker count (for example 7,500/s against 9,600/s with 8 workers): the batches per worker are too small |
| 2026-09-30 | First version of the rollout, before optimising | 1,750 decisions/s in one process; the encoders were re-deriving fleet arrivals for every route |
| 2026-09-30 | Recorded training data | about 1.5 KB per decision (was 4.7 KB before packing the legal-choice table) |
| 2026-09-30 | Model on one CPU thread, batch 48, after scoring only usable routes | 183 µs per decision (was 277 µs) |
| 2026-09-30 | Imitation run `imitation_v1`: 600 teacher games | 53 s to play; 241,444 decisions to learn from, 26,894 held out |
| 2026-09-30 | Same run: agreement with the teacher on held-out games | epoch 1: 67% · epoch 5: 78% · epoch 10: 80% · epoch 20: 86% (67% when the teacher sends, 99% when it waits) |
| 2026-09-30 | Same run: quick check after epoch 1 (16 games per opponent) | 94% against `aggressive_ai`, 97% against `careful_ai`, 6% against `strategic_ai` |
| 2026-09-30 | Same run: time | about 1 minute per epoch including the quick check; 30 minutes in all, with other jobs running alongside |
| 2026-09-30 | Same run: quick check against `strategic_ai`, epochs 11–20 | between 28% and 56%: 16 games is too few to rank neighbouring epochs |
| 2026-09-30 | Self-play from random weights (`selfplay_scratch_v1`): quick check against `aggressive_ai` / `careful_ai` / `strategic_ai` / its own starting point | before training 9 / 19 / 0 / 50% · round 10: 25 / 34 / 0 / 88% · round 20: 69 / 59 / 0 / 94% · round 30: 81 / 62 / 16 / 100% |
| 2026-09-30 | Same run: self-play games that end with a winner | 39% in round 1, 82% in round 30; average game 142 s down to 91 s |
| 2026-09-30 | Same run: entropy and policy change | entropy 4.6 down to 4.0; policy change about 0.009 per round, well under the 0.03 limit |
| 2026-09-30 | Same run: time | 39 minutes for 30 rounds (2,880 games), with other jobs running alongside |
| 2026-09-30 | First self-play run from the imitation model, stopped after 5 rounds: policy change per round | 3.07, 0.49, 0.35, 0.07, 0.05, against a limit of 0.03. Rounds 1–2 were meant to change only the value output |
| 2026-09-30 | Cause, measured on the imitation model: one value-only round at step size 3e-4 | moved the played moves' log-probability by 0.68 on average (40% of decisions by more than the clip range), with identical weights giving exactly 0 |
| 2026-09-30 | Same, with the moves held in place (weight 1) and step size 1e-4 | policy change 0.007, 5% of decisions past the clip range, value error 0.158 down to 0.099 |
| 2026-09-30 | Same, held but at step size 3e-4 | policy change 0.07: that step size is too large for a confident policy whatever the loss |
| 2026-09-30 | Self-play from the imitation model (`selfplay_from_imitation_v1`), corrected trainer: policy change per round | 0.011 in the two value-only rounds, then 0.011–0.018; the step size settled at 6.7e-5 |
| 2026-09-30 | Same run: value error against the "always saying 0" reference | 0.170 vs 0.325 in round 1; 0.075 vs 0.342 in round 30 |
| 2026-09-30 | Same run: entropy | 0.46 rising to 1.06: the entropy bonus is making a confident copy try more alternatives |
| 2026-09-30 | Same run: quick check against the model it started from | 50%, 53%, 69%, 41%, 88%, 59%, 62% at rounds 0 to 30: above even on average, but 16 games per point is noisy |
| 2026-09-30 | Same run: time | 38 minutes for 30 rounds |
| 2026-09-30 | League run `league_from_imitation_v1`: promotion matches (128 games each) | round 5: 52% · 10: 49% · 15: 59%, promoted · 20: 50% · 25: 60%, promoted · 30: 46% |
| 2026-09-30 | Same run: final ratings | `strategic_ai` 1403 · champion 2: 1324 · champion 1: 1264 · champion 0: 1233 · `careful_ai` 877 · `aggressive_ai` 720 |
| 2026-09-30 | Same run: time | 25 minutes for 30 rounds, promotion matches included |
| 2026-09-30 | Self-play from imitation, the same model evaluated twice against `strategic_ai` | 61.8% on the maps that overlap with its selection, 52.9% on fresh maps |
| 2026-09-30 | Self-play from imitation, final round (not selected) against `strategic_ai` | 43.4% on the standard maps (76 games), 57.9% on fresh maps (120 games): single results against one opponent swing by ± 10 |

Reading: the game engine is not the bottleneck. The network is, and it should
answer large batches on the GPU during training. One position at a time on
the CPU is fast enough for playing in the UI.

## Colab, and a crash at round 190 (2026-10-04)

**A crash:** `entropy_decay` stopped at round 190: after that round's
learning step the network gave NaN outputs, and the evaluation that follows
failed. The weights and optimizer state of round 189 were clean, and three
repeats of round 190's learning from there stayed finite (largest
probability shift 5.4), so it was a rare event, possibly a numerical glitch
on the Apple GPU. Learning now skips a batch whose loss or gradients are
not finite, and undoes a round whose weights end up broken
(`learning problems/...` in the metrics). The run resumed from round 189.

**Colab Pro, through VS Code** (the user's setup): Colab runs notebooks
only, on a runtime that starts empty. `ml colab build` writes a notebook
that carries the code (the game, its AIs and maps, `ml/` without runs and
tests; about 100 files) and any checkpoints in its first cell, unpacks
them, and runs the given commands, one cell each; files to send back
(e.g. a run folder) are packed into the last cell's output. After a run,
saving the notebook stores all of it in the file, and `ml colab read`
prints the output and unpacks the files. The trainers now pick an NVIDIA
GPU when there is one (`ml/device.py`), then the Apple GPU, then the CPU.

**`ml benchmark`** times one machine: the hardware, each network's forward
pass (one CPU thread; the GPU at 192 and 768 positions), one round of 96
games with CPU workers and with the GPU server, and one round of learning,
for 64x6 (trained), 128x8 and 256x8. First notebook:
`ml/colab/benchmark.ipynb`.
- [x] Colab's numbers (2026-10-04, `ml/colab/benchmark.ipynb`): A100-SXM4 80 GB, 12 cores
      (Xeon 2.2 GHz), 179 GB, Python 3.13, torch 2.11 (CUDA 13). Typical round = playing
      175,000 decisions (the trained 64x6's round) + 4 passes of learning:

| network | one CPU thread, 32 positions | A100, 192 / 768 positions | playing: 11 CPU workers / GPU server | learning | typical round |
|---|---|---|---|---|---|
| 64x6 | 22.3 ms (Mac 8.6–10.4) | 4.0 / 7.9 ms (Mac GPU 10–13 / 29–34) | 46 s / **31 s** | 6.2 s | **37 s** (Mac about 48) |
| 128x8 | 46.0 ms | 4.9 / 11.9 ms | 68 s / **33 s** | 8.6 s | **41 s** (Mac about 66) |
| 256x8 | 102.3 ms | 6.4 / 18.9 ms | 135 s / **33 s** | 13.7 s | **47 s** (Mac about 127) |

  - **On the A100 the GPU server pays off**, and playing time no longer
    depends on the network's size (31–33 s): the game simulation on the
    CPUs is the limit there, and Colab's cores are about 2.3 times slower
    than the Mac's (one thread: 22 against 10 ms). Learning is 4–5 times faster.
  - So a 256x8 network trains on Colab about as fast as the 64x6 on the
    Mac. Speed-up per round: 1.3x (64x6), about 1.6x (128x8), about 2.7x (256x8).
  - Google Drive mounts on the runtime (`/content/drive/MyDrive`), so long
    runs there can keep their folders on Drive and resume after a reset.
  - Next limit everywhere: the Python game simulation and encoding.
- [x] The Mac's numbers with the same command (2026-10-04, nothing else running,
      `ml/runs/logs/benchmark_mac.txt`):

| network | playing: 6 / 11 CPU workers | GPU server (6 / 11) | learning (Apple GPU) | typical round, Mac | Colab A100 |
|---|---|---|---|---|---|
| 64x6 | 20 / **15 s** | 40 / 41 s | 25.6 s | **40 s** | 37 s (1.1x) |
| 128x8 | 32 / **25 s** | 43 / 43 s | 39.9 s | **65 s** | 41 s (1.6x) |
| 256x8 | 60 / 55 s | **49** / 50 s | 73.7 s | **123 s** | 47 s (2.6x) |

  - **For the 64×6, Colab gives almost nothing** (37 against 40 s); it pays
    off for larger networks, where the Mac's learning half grows fastest.
  - **11 CPU workers beat 6 on this Mac** (15 against 20 s of playing for
    64×6): every Mac run so far left about a quarter of its playing speed
    unused. Future Mac runs: `--workers 11`. (The old measurement, "no gain
    past six", was made with a much smaller network.)

## A 256x8 network from random weights, on Colab (from 2026-10-04)

**The user's choice:** self-play only, no imitation and no kickstart: the
random-start route, at 20 times the size (4.5 million weights). Compared
with `deep_scratch`, the 64x6 network trained the same way for 500 rounds
(1263 at round 500), so the comparison changes two things at once: the size
and the entropy schedule.

**Settings** (`ml/scripts/colab_scratch_256.sh`): as `deep_scratch` (default
step size, 2 value-only rounds, the fading material hint), but the entropy
bonus starts at 0.01 and halves every 100 rounds towards 0.0005 (about
0.005 at round 100, 0.0017 at 300, 0.0008 at 500), and the networks are
served from the GPU (11 CPU workers, A100). 500 rounds, about 8-11 hours.

**How it runs** (`ml colab run`, notebook `ml/colab/scratch_256.ipynb`): the
notebook unpacks the code, mounts Google Drive, and starts the script
detached on the runtime. The script trains in 50-round chunks and copies
metrics, settings, latest, best and every 50th round to
`MyDrive/agent_game/runs/scratch_256` after each; on a new runtime it
carries on from there. A watch cell prints the latest line every 10
minutes (a running cell keeps the runtime from counting as idle). At the
end, `ml compare --gpu-server` rates rounds 0-500 of both runs on one scale
on Colab, and the last cell sends back the comparison, the metrics and the
final network (the 18 MB checkpoints stay on Drive). Tested locally end to
end with a tiny network and a folder in place of Drive, including carrying
on after the local copy was deleted.
- [ ] Results

## A third judge, from a stronger player's games (from 2026-10-04)

**Why this round is different:** judge1 (12,500 games) and judge2 (37,500)
both learned from games of deep_from_imitation round 500, and four times
the games played no better. Round 3 comes from today's best network,
`entropy_decay` round 300 (about +130 linked rating over round 500), so the
judge learns how games go between stronger players.

**The round** (`ml/scripts/judge3.sh`, started 2026-10-04 about 16:45):
1. Games `round3` (seed 2): 30,000 against itself, 1,500 each against
   aggressive_ai, careful_ai, strategic_ai, deep_from_imitation round 500
   and deep_scratch round 500; 8 positions per seat (2.9 KB each now).
2. Two judges, linear unit counts, random start: `judge3` on round 3 only,
   `judge3_all` on rounds 1-3.
3. The better one in play: their look-aheads head to head, 100 games.
4. Its full `judge-check` with the new player proposing moves (Atlas
   forecasts with round 500: new `--atlas-network`, because Atlas's frozen
   encoders cannot read newer networks).
5. The judge alone: Beast's own proposer (round 500) with the new judge,
   against Beast, 100 games.
- [ ] Results


## Acting once a second: an ablation (designed 2026-10-05)

**The question:** how much of the network's strength comes from acting on
every tick? The game takes one send per player per tick, so up to 10 a
second. This ablation allows at most one send a second, and asks what that
costs a network trained without the limit, and how much training under the
limit wins back.

**How often players send today** (48 fresh maps from 990,000, 24 for the
scripted AIs against themselves; a one-off script, 19 s):

| player | sends per second | ticks with a send | gaps between sends under 1 s | a 1 s limit would keep | seconds with any send |
|---|---|---|---|---|---|
| best network against itself | 7.2 | 72% | 99% | 12% | 93% |
| best network against strategic_ai | 7.2 | 72% | 99% | 13% | 96% |
| strategic_ai against itself | 4.0 | 40% | 95% | 18% | 77% |
| aggressive_ai against itself | 6.4 | 64% | 99% | 14% | 97% |
| careful_ai against itself | 0.9 | 9% | 76% | 37% | 39% |

"Best network": `shaping_entropy` round 150. "Would keep": of the sends in
these games, those at least 1 s after the last one kept, first come first
served (under the limit the games would of course go differently). The
limit bites hard: the network sends in bursts, on almost every tick, in
93% of all seconds. strategic_ai, the anchor of our scale, sends 2–4 times
a second, careful_ai less than once.

**What "once a second" means.** Two readings, which take away different things:
- **Cooldown (the main arm):** after a send, the player's next 10 ticks are
  forced waits; otherwise it decides on every tick as now. This takes away
  *how many* sends, not *when*: a player that waits can still act on the
  very tick something happens. It is the literal "at most one action a
  second", and the usual way to cap a game AI's actions (AlphaStar's
  actions-per-minute limits work like this).
- **Fixed rhythm (optional second arm):** the player decides only on ticks
  0, 10, 20, ... (the existing `decision_interval` setting at 10). This
  takes away the count *and* the timing: something that happens at tick 3
  is seen at tick 10, and a wait costs the whole second. The difference
  between the two arms is what choosing the moment within a second is worth.

A wait is not an action; only sends count. Nothing in the network's inputs
changes: under the cooldown the network is only asked when it may send, so
it never needs to know that it is cooling down, and that a send now rules
out another for a second is a fixed rule it can learn. The trainer already
measures time in ticks (discount and look-ahead per tick, `judge_moves`),
so decisions 10 ticks apart are credited on the same clock as decisions one
tick apart; nothing changes there either.

**Who is limited.** In training both seats are the network (self-play), so
both are limited: it learns the game in which everyone acts once a second.
In measurement only the network under test is limited; the panel plays as
always, so ratings stay on the usual scale, anchored on unlimited scripted
AIs, and every measurement is a handicap match. One extra reference: the
untrained limit on strategic_ai too, to see whether it hurts a hand-written
player as much as the network.

**Arms**, all starting from `shaping_entropy` round 150, so they start equal:

| arm | what | cost |
|---|---|---|
| Z: limited, untrained | round 150 played under the cooldown (the same network as C's round 0) | about 15 min of games |
| C: limited, trained | round 150 continues for 150 rounds of self-play under the cooldown, otherwise exactly `entropy_decay`'s settings (seed 1, step size 4.4e-5, no value-only rounds, no material hint, entropy 0.001 halving every 200 rounds towards 0.0002); 150 more if still rising | about 1.5–2.5 h, with nohup |
| control: unlimited, trained | **already run:** `entropy_decay` is this run without the limit, every round kept (1404 at round 0, 1410 at 100, 1394 at 300) | none |
| R (optional): fixed rhythm, trained | as C, with `decision_interval` 10 instead of the cooldown | as C |
| S (optional): limited, from random weights | as `deep_scratch` (64×6, 500 rounds, 1263) under the cooldown, on Colab | 6–9 h |

Arm S asks a different question: fewer, weightier decisions might make it
easier to tell which move won the game, so a random start could learn
*faster* under the limit.

**What is measured** (decided now, before any result):
1. Ratings: `ml compare` on the usual panel (strategic_ai, aggressive_ai,
   `deep_from_imitation` round 400, `deep_scratch` round 300), 100 games per
   match: C at rounds 0 (= Z), 50, 100, 150, and `entropy_decay` at the
   same rounds, on one scale. Limited players play limited, all others as
   always.
2. Head to head, 200 fresh games each: Z against round 150 unlimited (what
   the limit costs untrained); C round 150 against Z, both limited (what
   training under the limit won back); C round 150 limited against
   `entropy_decay` round 150 unlimited (what it still costs after adapting).
3. Behaviour (`ml behaviour`, 48 games): sends per second, units per send
   (new), sends to own bases, units away from the front, game length and
   decided games, and how often it waits when it may send (does it learn to
   keep its send for the right moment?).

**Reading the results** (also decided now):
- A large loss in Z and most of it back in C: acting on every tick is
  mostly habit, and the network can do with a seventh of its sends. That
  also answers part of the shuffling question (next steps): most of those
  sends were not needed.
- A large loss in Z and little back in C: speed is a real part of its
  strength.
- C level with or above the control: the limit helps learning; worth trying
  as a phase of training (limit first, then lift it).
- A small loss already in Z: the extra sends were nearly worthless.
With one seed per arm, a difference under about 50 rating points, or under
55% head to head, counts as no difference.

**Building it, one stage at a time:**
1. The rule. `Seat` gets `min_send_gap` (ticks; 0 = no limit, 10 = one send
   a second). The rollout does not ask a limited model seat while it cools
   down, and turns a limited scripted player's send into a wait (it is
   still asked, so AIs that remember things stay in step). A
   `--min-send-gap` option for self-play (both seats), and for `ml compare`
   and `ml behaviour` (naming the players it applies to); units per send
   and waits-when-free in the behaviour numbers. Tests: no two sends of a
   limited seat closer than the gap; an unlimited seat unchanged; games
   still replay exactly. Demo: one game, limited against unlimited, with
   each side's send ticks printed.
2. Arm Z and the strategic_ai reference (minutes); prediction 5 checked.
3. Arm C (a script in `ml/scripts/`, with nohup), its comparison and
   behaviour; prediction 6 checked.
4. Optional: arms R and S.

**The user's plan (2026-10-05):** train arm C only for now; later continue
from it with smaller gaps (5, 3, 2, 1 ticks) as separate runs, to see
whether play learned under the limit survives as the limit is lifted (the
hope: the limit forces it to pick high-value sends, and the habit stays).
The number to watch for that is "waits when it may send": if the network
still sends on nearly every tick it is allowed to, the rule did the
choosing and lifting it will probably bring the old pace back.

- [x] Stage 1: the rule (2026-10-05). `Seat.min_send_gap` in the rollout;
      `--min-send-gap` for self-play (the league refuses it). The networks
      of the run are limited in training and in the in-run evaluations,
      the starting model too, so "vs starting model" starts near 50% and
      shows what was learned under the limit; the scripted AIs are not.
      Checkpoints record the gap in their history. New number every round:
      `behaviour/waits when it may send`. `ml compare` and `ml behaviour`
      play a run's networks under the limit the run trained under (read
      from its `settings.json`), round 0 included, so round 0 of a limited
      run is arm Z; a panel checkpoint from such a run too; everyone else
      unlimited. `ml behaviour` also shows the gap, sends per second, units
      per send, waits when it may send, game length and decided games.
      The game's Neural AI does the same (`NeuralAgent(min_send_gap=...)`,
      per player), so `NEURAL_AI_CHECKPOINT=ml/runs/send_gap10/round_0150.pt
      ../.venv/bin/python main.py` shows it in the window as it trained.
      Tests: a limited seat's sends at least the gap apart, the other
      seat's not; ticks spent waiting out the gap not recorded; games
      still replay; seats get the gap in training and evaluation.
- [ ] Stage 2: arm Z (left for later; the in-run evaluation's round 0 is a
      first look: against strategic_ai 75% limited, against 87.5%
      unlimited for `entropy_decay`'s round 0, on the same 16 games).
      Behaviour, 48 games each (`ml behaviour`):

| | round 150, unlimited | the same, limited (Z) |
|---|---|---|
| sends per second | 7.3 | 1.0 |
| waits when it may send | 15% | 23% |
| units per send | 9.6 | 21.0 |
| sends to own bases | 78% | 72% |
| units away from the front | 33% | 39% |
| game length / decided | 101 s / 79% | 136 s / 56% |

      Untrained, its sends simply carry what piled up meanwhile (twice the
      units); the share to its own bases barely moves.
- [x] Stage 3: arm C (`ml/scripts/send_gap.sh`, run `send_gap10`, 20:22–21:16,
      about 21 s a round). Comparison `ml/runs/compare_send_gap10/` (the usual
      panel, 100 games per match); behaviour and two more head-to-heads in
      `ml/runs/logs/send_gap_behaviour.txt`.
- [x] Results (2026-10-05):

| round | 0 | 50 | 100 | 150 |
|---|---|---|---|---|
| limited (`send_gap10`) | 980 | 1075 | 1130 | 1159 |
| unlimited (`entropy_decay`, the control) | 1356 | 1366 | 1368 | 1364 |
| limited, against strategic_ai | 44% | 57% | 67% | 72% |

Head to head, fresh maps, both seats:
- Z (round 150 limited) against the same network unlimited: **19.5%** of
  200 (25–147, 28 draws).
- C round 150 against Z, both limited: **77.2%** of 200 (139–30, 31 draws),
  +7.7 standard errors.
- C round 150 limited against `entropy_decay` round 150 unlimited: **27%**
  of 100, −4.6 standard errors.

Behaviour (48 games each):

| | Z (untrained, limited) | C round 150 (limited) | control round 150 (unlimited) |
|---|---|---|---|
| sends per second | 1.0 | 0.9 | 7.0 |
| waits when it may send | 30% | 31% | 20% |
| units per send | 21.2 | 21.9 | 6.3 |
| sends to own bases | 72% | 65% | 78% |
| units away from the front | 39% | 37% | 34% |
| amount uncertainty (100%: even) | 59% | 37% | 59% |
| game length / decided | 138 s / 50% | 123 s / 67% | 96 s / 81% |

**Conclusions**
- **The limit costs a lot.** Untrained, −376 points (980 against 1356).
  Acting on almost every tick is a real part of this network's strength,
  not just habit.
- **Training under the limit wins back about half** (+179 in 150 rounds,
  77% against its untrained limited self), still rising but slowing
  (+95, +55, +29 per 50 rounds). It stays far below the control (27%
  head to head). Even so, one send a second beats strategic_ai 72%, which
  sends 2–4 times a second.
- **What it learned:** more decided amounts (uncertainty 59% → 37%), less
  shuffling (sends to own bases 72% → 65%, against 78% for the control),
  more games brought to an end (50% → 67% decided).
- **What it did not learn: to hold back.** It waits on 31% of the ticks it
  may send, as untrained (30%). The limit does the spacing, not the
  network, so lifting the limit should bring a much higher pace back at
  once. Whether the more decided amounts and the lower own-base share
  survive is what the smaller gaps test. (The per-round training number
  for waiting swings between 7% and 25%; the fixed-map measurement here is
  the one to trust.)
- **Larger sends come from the limit itself:** 21 units per send already
  untrained, against 6 unlimited; units pile up between sends.

**Lifting the limit again, without retraining** (2026-10-05; one-off
scripts, 200 games each on fresh maps, both seats): the network trained
under the limit, played with no limit at all, **beats the unlimited
control.**

| limited run's round, played unlimited | vs control round 150 | vs control round 300 |
|---|---|---|
| 50 | 61.0% (+3.1 SE) | 51.0% (+0.3) |
| 100 | 60.2% (+2.9) | 57.8% (+2.2) |
| 150 | 65.2% (+4.3); 57.2% on another 200 maps | 60.5% (+3.0) |

Round 150 against round 150 is 61% over 400 games (about +4.5 standard
errors), and against the control's round 300, which trained twice as
long, 60.5%: roughly +70 points over a control that had been flat
(round 300 against round 0: 54%). The gain against round 300 grows with
the rounds spent under the limit (51% → 58% → 60.5%).

Its behaviour unlimited (48 games): 6.8 sends a second (the old pace is
back at once, as expected), 75% to own bases (the control 78%, so the
shuffling did not stay down), 3.9 units per send (the control 6.3), and
amounts clearly more decided (uncertainty 46%, the control 59%; favourite
66% against 53%). So what carried over is not fewer or larger sends but
firmer amount choices. A guess at why: under the limit each send matters
ten times as much, so the learning signal on its amount is much stronger.

Caveats: one seed per arm, head to head against one opponent so far (not
yet rated on the panel), and several comparisons looked at.

**Next, agreed 2026-10-06:** (1) carry `send_gap10` on under the limit to
round 300 and judge every 50 rounds with the limit lifted (`ml compare
--unlimited`, new: no send limit for anyone), against the panel plus the
control's round 300; stop adding limited rounds when that stops rising.
(2) Then train without the limit from the best round, in one step.
(3) A second seed and the panel before it replaces Neural AI's network.
- [x] (1) (`ml/scripts/send_gap_more.sh`, 2026-10-06 08:26–09:27; resumed at
      round 151 with its own optimizer and schedule; the run's
      `settings.json` now says 300 rounds). Comparison with the limit
      lifted, `ml/runs/compare_send_gap10_unleashed/`, 100 games per match:

| `send_gap10` round, played unlimited | 150 | 200 | 250 | 300 |
|---|---|---|---|---|
| rating | 1469 | **1485** | 1485 | 1483 |
| against the control's round 300 (1404 here) | 61.5% | **64.0%** | 60.0% | 57.5% |
| against strategic_ai | 95% | 93% | 94% | 94% |

  **More rounds under the limit stopped helping after about round 200.**
  Played unlimited, the network sits about 80 points above the control's
  round 300 (1485 against 1404; the control's round 0, our former best,
  was 1404 on this panel too), which on the linked scale would be about
  1565 against 1487. Round 200 is the start for step (2): level with 250
  and 300 on rating, and the best against the control.
- [ ] (2) **Changed, the user's call (2026-10-06): lift the limit step by step.**
      From round 200, 50 rounds each at a gap of 9, 8, 7, 6 and 5 ticks,
      then look again (`ml/scripts/send_gap_ramp.sh`, started 11:55). Each
      step is its own run (`send_gap9` ... `send_gap5`), so every tool knows
      the limit a network trained under; weights carry over, the optimizer's
      memory starts afresh, the step size and the entropy schedule carry on.
      After each step its last round is rated with the limit lifted
      (`ml/runs/compare_ramp_gapN/`), on the panel plus the control's round
      300 and `send_gap10` round 200.
- [x] Results (11:55–14:07, 21–29 minutes a step; 100 games per match, every
      network unlimited):

| step's round 50 | vs the control's round 300 | vs `send_gap10` round 200 (the start) | vs strategic_ai |
|---|---|---|---|
| gap 9 | 56.5% | 51.5% | 92% |
| gap 8 | 57.5% | 52.5% | 93% |
| gap 7 | 58.0% | 52.5% | 94% |
| gap 6 | 59.0% | 57.5% | 94% |
| gap 5 | **68.0%** | 54.0% | 94% |

In training, at its own gap (round 50 of each step):

| gap | 9 | 8 | 7 | 6 | 5 |
|---|---|---|---|---|---|
| sends per game and seat | 115 | 135 | 153 | 159 | 177 |
| sends to own bases | 63% | 65% | 69% | 69% | 71% |
| waits when it may send | 38% | 33% | 21% | 23% | 29% |
| game length / decided | 112 s / 73% | 117 s / 77% | 113 s / 77% | 103 s / 82% | 99 s / 80% |

- **Lifting the limit gradually kept the gain** and perhaps added a
  little: against the start 51.5–57.5%, 53.6% pooled over the five steps
  (500 games, about +1.6 standard errors); against the control 56.5% rising
  to 68% at gap 5 (+3.6 standard errors on its own).
- **Shuffling creeps back as the limit loosens:** sends to own bases 63% at
  gap 9, 71% at gap 5 (the unlimited control: 78%).
- **Careful with these comparisons' ratings** (1424, 1436, 1449, 1462, 1476):
  with one network per comparison, every panel member is rated only
  through its games against that network, so the network's rating is in
  effect its score against strategic_ai (92% → 94%). Only the head-to-head
  scores carry information. A comparison of several networks at once is
  needed for real ratings.

## Ratings for the game's menu (2026-10-05)

**What:** the home screen now shows each AI's estimated rating on our scale
(strategic_ai = 1000), read from `ais/ratings.json`; without the file it
shows none. `ml bot-ratings` (`ml/evaluation/bot_ratings.py`) writes it.
The quick players (the three scripted AIs and Neural AI, rated as
`deep_from_imitation` round 500, the network long scripts leave in
`ml/runs/latest.pt`) play each other here, 100 games per pairing on fresh
maps from 980,000. Atlas, Atlas Clean and Beast take minutes per game
against a network, so their matches are the ones recorded earlier (Beast,
the two-move look-ahead, move scoring, Atlas Clean, Atlas as a yardstick;
17 matches listed in the module), joined to the scale through the players
that met both sides then. All fitted at once, as in `ml compare`.

| AI | rating | from games |
|---|---|---|
| Beast | 1473 | 120 |
| Neural AI (round 500) | 1332 | 520 |
| Atlas | 1263 | 208 |
| Atlas Clean | 1137 | 200 |
| Strategic AI | 1000 | 432 |
| Careful AI | 777 | 372 |
| Aggressive AI | 654 | 372 |

Consistent with earlier numbers: round 500 is 1357 on the linked scale,
Atlas Clean was put at "roughly 1150–1180". Beast's and Atlas's ratings
rest on 60–200 games, so about ±50. When Beast changes (`ml beast set`),
its matches have to be measured again; `ml bot-ratings` warns when Beast
is no longer the player it rated.

## Amount options that send the same units (2026-10-08)

**The user's point:** the eighths are rounded up (`max(1, ceil(fraction ×
units))`), so a base with 2 units offers 1, 1, 1, 1, 2, 2, 2, 2: eight legal
options, two different moves. The entropy bonus, the PPO probabilities and
the policy-change limit all count options, not moves, so spreading over
options that play identically looks like being undecided, earns the bonus,
and uses up the allowed policy change without changing play. (The same
flaw as in "Teaching the policy from the look-ahead", there fixed only
for the teaching targets.)

**Measured** (8 games each on maps 950,000+, every send of both seats;
script kept as `ml/runs/logs/amount_duplicates_script.py`):

| network | sends | units at the source: median / ≤3 / under 8 | different amounts among the 8 eighths | amount entropy over options / over moves |
|---|---|---|---|---|
| control `entropy_decay` r300, unlimited | 8,848 | 4 / 47% / 64% | 4.5 | 1.17 / 0.69 |
| `send_gap10` r200, at gap 10 | 2,055 | 11 / 24% / 42% | 6.1 | 0.74 / 0.53 |
| `send_gap10` r200, unlimited | 10,477 | 2 / 62% / 74% | 3.6 | 0.92 / 0.35 |
| `send_gap5` r50, unlimited | 8,090 | 2 / 64% / 79% | 3.3 | 0.71 / 0.26 |

- **The degeneracy is the normal case in unlimited play:** half the sends
  leave a base with 3 units or fewer, and the eight eighths offer only 3–4
  different amounts. 40–60% of the measured amount entropy is spread over
  options that send the same units.
- **A likely explanation of the send-limit result:** under the limit, bases
  hold about 11 units, the eighths are mostly different moves, and the
  amount choice gets a clean signal. The networks from the limited runs are
  much more decided over real moves (0.26–0.35 against the control's 0.69).
- **Earlier numbers are inflated:** every "amount uncertainty" in this file
  (for example "nearly random, 95% of an even pick" at entropy 0.01) was
  measured over options, duplicates included.

## Sends between own bases: relays, not shuffling (2026-10-08)

**Question (the user's):** does the network understand where the front is,
when a send can only cross one route and reaching the front from the rear
takes several sends? Every send of 48 self-play games per network (maps
940,000+) was replayed in its position; a send to an own base counts as
forward when its target is closer (shortest travel time) to the nearest
enemy base than its source. Script kept as `ml/runs/logs/front_moves_script.py`.

| network | sends to own bases | of those forward / backward / level | landing on a front base | sends leaving a front base |
|---|---|---|---|---|
| control `entropy_decay` r300 | 79% | 78% / 21% / 1% | 67% | 56% |
| `send_gap5` r50, unlimited | 77% | 82% / 17% / 0% | 66% | 60% |

("Front base": has a neighbour that is not mine.)

- **The "70% shuffling" was mostly relaying.** Four in five sends between own
  bases move units toward the enemy, and two thirds land on a front base.
  The network finds the front; moving units there takes one send per route.
- About one in five moves units away from the nearest enemy base. Some of
  that may be defence of a base behind the front; not looked at further.
- Consequence for the new feature set: the front needs no special help to be
  found, but two cheap inputs make it explicit (distance to the nearest
  enemy base per base; how much closer to it a route leads).

## A new feature set and free amounts (from 2026-10-08)

**The user's design.** Inputs: counts as plain numbers instead of logarithms
(divided by a reference: units / 100, production / 10, so they stay near 0
to a few); the forecast "if nobody sends anything new" at 5, 10, 20, 50 and
80 ticks (owner as three 0/1, plus units) and after every fleet in flight
has landed; whether a base has a neighbour that is not mine; and, after
"Sends between own bases", the distance to the nearest enemy base and, per
route, how much nearer the enemy it leads (`forward`). Pair features and
the four global features unchanged. Per route also the capture cost as a
plain count, since with free amounts the network has to hit it exactly.
Output: wait or a route as now, then any whole number of units instead of
nine options, from a blend of curves per route (no repeated options by
construction). Start: copy our current best first (kickstart), with
imitation or a random start as later comparisons.

**Stages:** (1) the inputs, (2) the free-amount output and its maths (pick,
probability, entropy), (3) the trainers, (4) the experiment.

- [x] Stage 1, the inputs (2026-10-08). Appended to the lists, so every saved
      network still loads and reads what it did: 27 base columns (41 in
      all) and 2 route columns (5). `FEATURE_SETS` names what a network
      reads: "v1", the first 14 / 3 / 0 / 4 (all our networks so far), and
      "v2", 34 base, 3 pair, 2 route, 4 global. `ml show-state --features
      v2` (the default now) prints them, the forecast as a table of owner
      and units per horizon. A network built with default settings reads
      only the columns from before (`COLUMNS_BEFORE_V2`); a v2 network
      will choose its columns by name (stage 3).
  - Tests: every horizon matches the game played on without new sends
    (over 500 base and horizon checks); distance to the enemy and
    `forward` against shortest paths worked out separately in the test;
    the front flag; the plain counts against the log ones; renamed and
    shuffled bases, swapped colours and mirrored starts give the same
    columns. All 309 + 69 tests pass.
  - Cost: encoding a position takes 111 µs instead of 62; on real
    self-play (32 games, one process) 12.6 s against 11.6 s, about 9%.
  - To keep in mind: early in a game the plain counts are small (3–7
    units are 0.03–0.07); the network's first layer can scale them up.
    Rollouts record every column, so a recorded position is about three
    times larger (41 base columns instead of 14); recording only the
    columns the playing networks read would avoid that (stage 3).
- [x] Stage 2, the free-amount output (2026-10-08). `ml/models/amounts.py`:
      per route a blend of 3 logistic curves (weight, centre, width each);
      the probability of k units is the blend's area between k - 1/2 and
      k + 1/2, the edges taking the tails, so 1, ..., most add up to one and
      no two choices send the same units. Each centre is fraction × most +
      share × needed + offset: the network picks proportions, the exact
      numbers come from the legal-move table, which now has `most` (units
      at the source) and `needed` (what a capture takes) per route. The
      curves start on "just enough", "half" and "everything". Narrowest
      width 0.05 units (99.99% on one amount). Entropy is exact over every
      amount up to 256 units; above that the rest is lumped into one cell.
      `graph_transformer` takes `amounts=free` (and `amount_curves`); the
      policy maths (pick, probability, entropy), the agent and the
      encoding of teacher moves (exact units) handle both kinds.
      `ml show-actions --network free|PATH` shows P(wait), P(route) and
      the likeliest amounts per route (an option network's options added
      up by units).
  - Seen with an untrained network: where a capture is affordable, "just
    enough" and "everything" can coincide (5 of 5 units: 52%); where it
    is not (needs 5, has 3), the mass gathers on everything (67%); on
    routes to own bases (nothing needed) the "just enough" curve sits on 1
    unit. It played 300 ticks with no move rejected.
  - Tests: probabilities over every choice (wait, route, every amount)
    sum to one and the entropy equals its definition over that list;
    draws follow the distribution; the anchors hit exact amounts; far-off
    curves keep finite values and gradients; only routes an option network
    may use are used; base order does not matter; greedy takes the most
    likely; every amount 1..most is accepted by the game. All 330 + 69
    tests pass.
  - Not yet: rollouts, recorded decisions, the trainers and the GPU
    inference server still assume options (stage 3).
- [x] Stage 3, the trainers (2026-10-09). Everything that plays or learns
      now takes a network with free amounts and the v2 inputs:
  - **Recorded games** keep, per decision, the units each send sent and,
    per route, four numbers: most, needed, hold and surplus. Every amount
    option is worked out from these four (`option_units`, now the one place
    the options are defined; legal moves take 65 µs instead of 60), so a
    recorded game can rebuild what each option would have sent. A free
    network's choice is its units, and its trajectory says so.
  - **Inference:** the GPU server's shared memory carries most and needed,
    and the server draws the units from the curves on the device; the
    workers decode them as units.
  - **Decision sets** for a free network learn the units and give each
    batch the most and the needed of every route. A set for an option
    network refuses choices made with free amounts.
  - **The network reads its inputs by name:** `--set feature_set=v2` (with
    `amounts=free`): the sizes follow the set and are saved with it, so a
    set must never change once a network is trained on it.
  - **Imitation:** a free network learns the teacher's exact units.
  - **Kickstart:** a free student learns, per route, the teacher's
    probability for each number of units, options that send the same units
    added together (the divergence over those amounts). Teaching from a
    free teacher is not built.
  - **Self-play** needed no new maths: PPO works on the probability of the
    chosen units, and the entropy is the one from stage 2.
  - **`ml behaviour`:** amount uncertainty over every whole number from 1
    to most, and hold / surplus counted when exactly those units are sent.
  - **Not for free networks yet:** the look-ahead (Beast, `value:` and
    `lookahead:` players), `ml mistakes`, `ml show-model`; each says so.
  - **Left out on purpose: recording only the columns a network reads.**
    Measured: 24 self-play games record 160 MB (about 640 MB for a round
    of 96), up from about 60 MB before stage 1. A v2 network reads 34 of the
    41 base columns, so for it this would save about 15%; only the old
    networks (14 columns) would gain much. Not worth the extra machinery.
  - Tests: free networks played here and through the server: every move
    accepted, units from 1 to everything, recorded probabilities equal to
    what the network says; the recorded limits rebuild every position's
    options exactly; decision sets; the v2 network reads exactly its
    columns (changing any other changes nothing) and survives saving; the
    kickstart divergence against a direct calculation in double precision
    (and some routes do merge options); a self-play update makes amounts
    judged good more likely; small runs of imitation, kickstart and
    self-play (through the server) with a free network. All 347 + 69 tests
    pass.
  - **Demo** (`ml/scripts/free_demo.sh`, 09:23–09:28, log
    `ml/runs/logs/free_demo_log.txt`): a fresh 64×6 v2 network with free
    amounts, kickstarted from `send_gap5` round 50 on 96 teacher games for 4
    epochs (the default is 1200 games and 20), then 5 rounds of 48
    self-play games. It all runs. Kickstart: held-out divergence 2.82 →
    1.76, the teacher's favourite first step 9% → 36% (the option
    student `large_kickstart` reached 0.38 and 62% on 12 times the games).
    Self-play: value error 0.10–0.13 against 0.22–0.32 for always saying 0,
    entropy about 2.2. Scores on 8 games each: before self-play strategic_ai
    25%, aggressive 38%, careful 50%; after 5 rounds 12%, 50%, 44%, and 38%
    against its start. A short kickstart makes a weak player; this shows
    the chain works, not what the design can do.
  - `ml behaviour` (24 games; the teacher under its gap of 5, the free
    networks unlimited): the free networks send 7.5 a second, 16 units a
    send, 83% to own bases; amount uncertainty 73% of an even pick over
    every whole number (teacher: 29% over its options), favourite amount
    59–61% (teacher 77%). They send exactly the "hold" units 20–22% of the
    time it is on offer and "surplus" 10–12%; with free amounts that counts
    any send of that size, so it is not the same measure as an option
    network choosing the option.
  - Speed: a free 64×6 network's self-play takes about 1.7 times as long per
    game (39 s per 48 games against about 45 s per 96 for `deep_scratch`).
    Not profiled yet.
  - For stage 4: the amount's entropy is on a larger scale now, up to
    log(most) per route instead of log(11), so the entropy bonus pushes
    harder on amounts than before.
- [x] Stage 4, first look (the user's plan, 2026-10-09): self-play only from
      random weights, 100 rounds, then rated next to `deep_scratch` (the old
      design, trained the same way) at rounds 0, 50 and 100, both rounds 100
      head to head. No control arm and no speed profiling first, the user's
      call. Run `free_scratch` (`ml/scripts/free_scratch.sh`, started 09:33):
      64×6, v2 inputs, free amounts; settings as `deep_scratch` (seed 0, so
      the same training maps; 2 value-only rounds; the material hint fading
      over 50 rounds), no send limit, except the entropy bonus: 0.001
      halving every 200 rounds towards 0.0002 (agreed with the user;
      `deep_scratch` had 0.01 throughout). So the comparison changes the
      entropy bonus as well as the design.
      - **A memory problem, found and fixed (11:55).** Rounds slowed from
        about 95 s (rounds 1–10) to 5–12 minutes (rounds 16–28), although the
        games got shorter. The training process had grown to 31 GB on the
        16 GB machine (22 GB of small allocations, 7.4 GB on the GPU) and the
        machine was swapping 25 GB. Cause: the same Apple GPU problem as in
        the first sweep ("A problem found on the way"). The entropy bonus for
        free amounts, new in stage 2, worked on two sizes that changed with
        every batch: the number of usable routes, and the grid width (the
        batch's largest `most`, up to 256). Kickstart's free-amount divergence
        had the first of the two as well.
      - Fix: route lists outside the network are padded to a few lengths, as
        the network's own list already was (`padded_length` and
        `readable_route_list` in `ml/models/interface.py`, shared by all
        three). Each route's amount entropy is worked out on the smallest of
        five fixed grids (16, 32, 64, 128 and 256 cells) that holds it, and
        each group is padded too. The values do not change; a new test checks
        values and gradients against one wide grid. Measured with 17,000
        routes per batch (the real count, from 24 games of round 28; 88% of
        routes have at most 16 units, while a batch's largest `most` is about
        230): memory stays at 1.51–1.56 GB over 150 learning steps, where
        before it grew by 5 MB a step, and a step takes 34 ms instead of
        113 ms.
      - The run was stopped after round 28 and resumed from it at 12:12.
        Rounds now take about 60 s (rounds 1–10 took about 95 s, as the old
        entropy grid was three times the work; `deep_scratch` took about
        40 s). The process held 5.8 GB after the first resumed round and grew
        by about 100 MB per round over the next four (it restarts after round
        78 anyway).
      - **Results** (training finished 13:19, `ml compare --name free_scratch`
        13:24; 100 games per match, strategic_ai fixed at 1000; one seed each):

        | | free amounts, v2 | options, v1 (`deep_scratch`) |
        |---|---|---|
        | rating, round 0 | 526 | 512 |
        | rating, round 50 | 971 | 902 |
        | rating, round 100 | 1122 | 1108 |
        | round 100 vs strategic_ai | 76% (74 W 4 D 22 L) | 61.5% (56 W 11 D 33 L) |
        | round 100 vs aggressive_ai | 93% | 91% |
        | round 100 vs careful_ai | 93.5% | 91% |
        | round 50 vs strategic_ai | 46% | 33.5% |
        | round 100 head to head | 44% (41 W 6 D 53 L, −1.2 SE) | 56% |

      - Read: the new design learned at least as fast per round. It was ahead
        of the old one against the fixed panel at round 50 and at round 100
        (against strategic_ai by about 14 points, about two standard errors at
        100 games each). But the two rounds 100 played each other about even,
        and the old design slightly ahead (−1.2 standard errors, not a real
        difference). So it beats the scripted players more often without
        beating the old network: in ratings the two are 14 points apart, which
        is nothing.
      - It became far more decided than `deep_scratch`: entropy 1.93 at round
        100 against 3.68, falling from 3.5. That is also the smaller entropy
        bonus (0.001 halving, against 0.01), so this run cannot say which of
        the two did it. Its self-play games got shorter (69 s against 98 s at
        round 100) and more often decided (97% against 83%).
      - Speed: about 56 s per round after the memory fix, about 1.4 times
        `deep_scratch`'s 40 s. The 3.5 hours in the report are inflated by the
        memory problem; without it, 100 rounds take about 1.6 hours.
      - Not a new best (the best networks are far above 1122 on their scales),
        so Beast stays as it is.
- [x] Stage 4, continued (the user's plan, 2026-10-09): `free_scratch` trains
      on to round 300, as the same run (`ml/scripts/free_scratch_more.sh`,
      started 13:45): its round count raised in `settings.json`, then
      `--resume`, so weights, optimizer, step size and the entropy schedule
      carry on (the bonus goes from 0.00077 at round 100 to 0.00048 at round
      300). `deep_scratch` already went to round 500, so it needs no more.
      Then `ml compare` at rounds 100, 200 and 300 of both, and both rounds 300
      head to head. Round 101 carried on from round 100 as expected (step
      size 1.0e-04, entropy 1.86 after 1.93).
      - **What the memory that builds up actually is** (measured 2026-10-09,
        the user's question; learning steps of the real 64×6 free network on
        the GPU with self-play-shaped batches, 1024 positions, maps of 4 to 29
        bases). The earlier conclusion, "memory that is never given back", was
        mostly wrong; it had been concluded without trying to give it back.
        - Most of it is PyTorch's cache of freed GPU buffers. Batches from
          different map sizes (26 sizes among the training maps) and route
          counts need buffers of different sizes, and the cache keeps them
          all. `torch.mps.empty_cache()` returns it: 3.4 GB down to 0.5 GB,
          the GPU part from 2.6 GB to 60 MB and the CPU part from 650 to
          260 MB. In the live run at round 110, 4.2 of the 6.3 GB is this
          GPU cache.
        - What is really kept is about 0.6 MB per new shape (programs
          compiled for each shape): 32 MB for 52 new shapes, then flat.
        - `gc.collect()` changes nothing: it is not Python objects.
        - Releasing after every learning step makes a step 2.3 times slower
          (300 against 690 ms). Releasing once per round would cost almost
          nothing.
        - Built (the user's call): `release_cached_memory` in `ml/device.py`,
          called once per round after learning in self-play (and so the
          league), and once per epoch in kickstart, imitation and the judge's
          training. A test checks that it gives freed GPU memory back. The
          running `free_scratch` picks it up when its next process starts at
          round 151, which is the check in a real run: memory per round in
          rounds 101–150 (without) against 151 onwards (with).
        - Changed on the user's suggestion: round 133 took 4 minutes (the
          machine was swapping, 14.9 of 15.4 GB of swap used, our process at
          7.9 GB), so the run was stopped after round 134 and resumed at 14:19
          with the release. Rounds 117–134 without it: 6.6–7.9 GB, of which
          4.2–5.3 GB was the GPU cache. Rounds 135–137 with it: 1.4–2.1 GB,
          of which 26–67 MB on the GPU; rounds take 50–70 s again.
        - Over 31 rounds with the release (135–165) the process stayed at
          1.6–2.1 GB with no trend, the GPU part at 26–67 MB. Only the small
          CPU allocations crept up, by about 2 MB a round (189 to 252 MB),
          which would be about 2 GB after 1000 rounds. So long runs no longer
          need a fresh process every 50 rounds; `--stop-after` stays for
          splitting a run, and the running script keeps it (harmless).
      - **Paused after round 250** (16:30, the user's call), to be resumed
          later with `nohup ml/scripts/free_scratch_more.sh >>
          ml/runs/logs/free_scratch_log.txt 2>&1 < /dev/null &`, which carries
          on from round 251 and then runs the comparison.
      - Rounds got slower with the release, 53 s per round (102–132) against
          65 s (136–184) and 72 s (236–250). Games also got longer (71 to 78
          s), but rounds 30–78 had games as long (82 s) at 56 s per round.
          Measured (the same synthetic learning rounds as above, 26 map sizes,
          52 steps a round): learning took 9.4 s a round keeping the cache,
          12.2 s releasing it every round (+30%: each shape's buffers are
          allocated afresh in the next round), and 9.4 s releasing only above
          4 GB. During learning the GPU holds about 3.3 GB either way, so
          releasing every round only lowered memory between learning phases,
          not at the peak.
      - So the release now waits until the GPU holds more than 4 GB
          (`CACHE_LIMIT` in `ml/device.py`; `limit=0` always releases). The
          run was resumed with it from round 251 at 17:12.
        In the run, rounds 252–262: 49.5 s per round (games 76 s), against
          72 s for rounds 236–250 (games 78 s) with the release every round,
          and 53 s for rounds 102–132 without any. The process held 5.4–5.7 GB
          (about 3.2 GB of it on the GPU) and the cache was given back about
          every four rounds (after rounds 253 and 257), so memory has a
          ceiling instead of growing.
        Over all of rounds 252–300, 53 s per round (median).
      - **Results** (training finished 17:56, `ml compare --name
        free_scratch_300` 17:59, `ml/runs/compare_free_scratch_300/report.txt`;
        100 games per match, strategic_ai fixed at 1000; one seed each):

        | | free amounts, v2 | options, v1 (`deep_scratch`) |
        |---|---|---|
        | rating, round 100 | 1151 | 1089 |
        | rating, round 200 | 1245 | 1156 |
        | rating, round 300 | 1289 | 1193 |
        | vs strategic_ai, rounds 100 / 200 / 300 | 72% / 80% / 86% | 60% / 71% / 74% |
        | round 300 vs aggressive_ai | 97% | 94.5% |
        | round 300 vs careful_ai | 94% | 93.5% |
        | round 300 head to head | 63% (61 W 4 D 35 L, +2.6 SE) | 37% |

      - Read: the new design pulled ahead. Rounds 100–300 added +138 to it and
        +104 to the old one, and the gap grew from 62 points at round 100 to 96
        at round 300. Head to head turned from 44% at round 100 (−1.2 SE) to 63%
        at round 300 (+2.6 SE), so it now beats the old network itself, not
        only the scripted players.
      - The same rounds 100 were rated 1122 and 1108 in the first comparison,
        1151 and 1089 here: single ratings are good to about ±30, as before.
      - Still rising, more slowly: +94 over rounds 100–200 and +44 over
        200–300 (old design +67 and +37). Entropy 1.93 at round 100, 1.58 at
        200, 1.55 at 300; the bonus went from 0.00077 to 0.00048.
      - Against the old design's later rounds, roughly: in the Stage 10
        comparison `deep_scratch` was 1207 at round 300 and 1263 at round 500.
        Here its round 300 is 1193, so `free_scratch` round 300 is about 1300
        on that scale: above `deep_scratch` round 500 after 300 rounds, below
        `deep_from_imitation` round 500 (1357, Neural AI's network). Two
        scales linked by one checkpoint, so only a rough placing.
      - Time: 53 s per round against `deep_scratch`'s 40 s (1.3 times). The
        report's 7.0 hours include the memory problem and the swapping;
        without them 300 rounds take about 4.5 hours, against 3.3.
      - Caveats: one seed, and the entropy bonus differs from `deep_scratch`'s
        (0.01 throughout), so this shows the new design with the smaller
        bonus, not which of the two made the difference.
      - Not a new best: Neural AI's network is still above it, and Beast cannot
        use a free network yet. Beast stays as it is.
- [ ] Stage 4, the entropy bonus (the user's plan, 2026-10-10): which of the two
      made `free_scratch` better, the design or its smaller bonus? Taken from
      the other side: the new design with the old bonus. Run
      `free_old_entropy` (`ml/scripts/free_old_entropy.sh`, started 11:40):
      `free_scratch` exactly (the same start `free_v2_seed0.pt`, seed 0 so the
      same training maps, the same settings), except an entropy bonus of 0.01
      throughout, as `deep_scratch` had. 300 rounds, then `ml compare` at
      rounds 100, 200 and 300 of all three runs, and the three rounds 300 head
      to head.
      - What it can tell: if it lands near `free_scratch`, the design did it; if
        near `deep_scratch`, the bonus did. A free amount's entropy runs up to
        log(most) per route instead of log(11), so 0.01 pushes harder on the
        amounts here than it did on the options; it may spread the amounts out
        more than it did for `deep_scratch`.
      - Settings checked against `free_scratch`: only `entropy_weight` (0.01
        against 0.001), `entropy_half_life` (0 against 200) and
        `entropy_floor` differ. Rounds 1–3: about 65 s each, games 126–141 s
        (the start plays long games), entropy 3.47 → 3.29.
      - **Paused after round 124** (13:56, the user's call, to look at the
        entropy's parts and the lookahead first). Resume with `nohup
        ml/scripts/free_old_entropy.sh >> ml/runs/logs/free_old_entropy_log.txt
        2>&1 < /dev/null &`; it carries on from round 125, then compares.
        At round 100 against `free_scratch`: waits 3.3% of the time it could
        send (9.8%), 773 sends per game and seat (524), games 93 s (69 s), 86.5%
        decided (97%), entropy 3.0 (1.93).
- [x] The entropy in its two parts (the user's plan, 2026-10-10, after an
      outside review). The bonus rewards the entropy of the whole choice: which
      route (or waiting), plus the amount's entropy for each route weighted by
      how likely the route is. Through the second part, the bonus works like an
      extra reward of `bonus × H(amount | route)` for choosing a route, and
      waiting has none. With options every route had the same most (about
      log 9), so this pulled towards sending at all; with free amounts the most
      is log(units available), up to log 256, so it also pulls towards routes
      from bases with many units, and harder. And the old design's entropy
      counted duplicate options (40–60% of its amount part), so 1.93 against
      3.68 (`free_scratch` against `deep_scratch` at round 100) overstates how
      much more decided the new design is.
      - Built: `entropy_parts` in `ml/models/policy.py` (`entropy` is their
        sum, unchanged); self-play logs `entropy/of the route choice` and
        `entropy/of the amounts` next to `entropy/of the policy`. Tests: the
        route part against the route choice's own entropy and the parts adding
        up, for both designs (the whole was already checked against every
        choice); a position certain to wait has no amount part; self-play's
        logged parts add up.
      - Measured on saved rounds (32 self-play games each on fresh maps,
        seeds 50,000+; mean per decision; "per send" is the amount part divided
        by the chance of sending):

        | | route | amounts | amounts per send | chance to send |
        |---|---|---|---|---|
        | `free_scratch` 50 / 100 / 200 / 300 | 2.31 / 1.83 / 1.62 / 1.43 | 0.81 / 0.64 / 0.54 / 0.64 | 0.87 / 0.71 / 0.59 / 0.68 | 93 / 90 / 92 / 94% |
        | `free_old_entropy` 50 / 100 / 124 | 2.50 / 2.41 / 2.19 | 1.42 / 1.34 / 1.23 | 1.48 / 1.39 / 1.29 | 95 / 96 / 95% |
        | `deep_scratch` 100 / 300 (options) | 2.27 / 2.13 | 1.92 / 1.81 | 1.97 / 1.88 | 97 / 96% |

      - Read: with the same bonus (0.01) the route choice is as undecided in
        both designs (2.41 and 2.27 at round 100); the gap in the total was the
        amounts (options nearly even over about 7 of 9 choices, e^1.97). The
        10 times larger bonus doubled the amount part (0.64 → 1.34, about 2
        against 4 amounts in play per send) and raised the route part by a
        third. The pull towards routes with uncertain amounts is bonus × the
        amount entropy per send: about 0.013 per decision at 0.01, about 0.0005
        at `free_scratch`'s 0.00077 (advantages are scaled to 1 per round).
        So at `free_scratch`'s bonus the side effect is small.
      - `free_old_entropy` learned more slowly: against strategic_ai in the
        16-game checks of rounds 60–120, 44% on average against 67% for
        `free_scratch` (7 checks each).
      - For the lookahead, from the 32 games of `free_scratch` round 300:
        sends land after 2.6 s (median; 90% within 4.2 s), so even 0.99 credits
        a send's landing directly (72%). Games last 98 s (median; 10% reach
        the 180 s limit). The value output explains 95–96% of the result in
        the last 20 s of a game, 78% at 20–40 s, 66% at 40–60 s and 21%
        earlier (part of that is the game still being open, not the value
        output's fault). Advantages (before scaling) by lookahead:

        | lookahead | credit half-life | weight at 10 / 30 / 60 s | spread | agreement with 1.0 |
        |---|---|---|---|---|
        | 0.99 | 6.3 s | 33 / 3.6 / 0.1% | 0.131 | 0.69 |
        | 0.995 | 11.5 s | 55 / 16.5 / 2.7% | 0.161 | 0.82 |
        | 0.998 | 23 s | 74 / 41 / 16.5% | 0.202 | 0.94 |
        | 1.0 (result only) | 69 s | 90 / 74 / 55% | 0.278 | 1 |

        The variance grows 1.5 times at 0.995, 2.4 at 0.998 and 4.5 at 1.0
        (against 0.99); 0.998 is close to judging by the result alone.

## Curves in TensorBoard (2026-10-10)

The pygame monitor has no smoothing, no log scale and shows one run at a
time. `ml tensorboard` (`ml/training/board.py`, the user's call) shows the
same metrics in TensorBoard, in the browser: every run's `metrics.jsonl` is
copied into TensorBoard's format in `ml/runs/tensorboard/` (rebuilt at each
start, holding only the runs asked for) and kept up to date every 5 s, so a
running run grows live. Nothing in the trainers changes, and every old run
can be shown. Step is the round; "Relative" is hours of training.

- Tried on `free_scratch` and the running `free_old_entropy` (18 charts each,
  all 300 rounds of `free_scratch`; a new round of `free_old_entropy`
  appeared within one check).
- On the way: TensorBoard keeps a random 1000 points per line unless told
  otherwise, and `--samples_per_plugin scalars=0`, "keep all" in its help,
  kept none with its fast loader (which is installed). The limit is 100,000.
- Tests: `ml/tests/test_board.py` (charts by round and training time, values
  that are not numbers left out, a half-written line waits, a run started
  again from scratch replaces its copy).

## Playing the favourite instead of drawing by chance (2026-10-10)

**Question (the user's, from watching Neural AI shuffle units):** how much of
the shuffling is the network drawing a move by chance, and how much is the
move it likes best? Every game so far (training, the checks, `ml compare`,
Neural AI in the game) draws by chance. Large-bonus networks play their
favourite move only about a third of the time (35% for `deep_scratch` round
300, 54% for `free_scratch` round 300).

**Measured** (`ml/runs/logs/greedy_shuffling_script.py`, output
`greedy_shuffling.txt`; 100 games per line, maps 960,000–960,049, both seats;
every send classified when it is made, as in "Sends between own bases"; "sent
back" = the same player sends from B to A within 5 s of a send from A to B):

| | score | median game | own-base sends backward | own-base sends sent back within 5 s | backward, of all sends |
|---|---|---|---|---|---|
| Neural AI's network, by chance, vs strategic_ai | 88.5% | 81 s | 32% | 55% | 24% |
| Neural AI's network, favourite, vs strategic_ai | 88.5% | 49 s | 21% | 34% | 14% |
| `free_scratch` r300, by chance, vs strategic_ai | 82.0% | 62 s | 23% | 42% | 16% |
| `free_scratch` r300, favourite, vs strategic_ai | 81.0% | 57 s | 21% | 33% | 14% |
| Neural AI's network head to head: favourite | 66.5% | 74 s | 26% | 48% | 20% |
| … against itself by chance | 33.5% | | 35% | 70% | 28% |

- **For Neural AI, playing its favourite is better.** It beats its by-chance
  self 66.5 to 33.5 (+3.5 standard errors) and wins against strategic_ai
  just as often, 40% faster. No loops: it shuffles less, not more.
- **About a third to two fifths of the shuffling was chance.** Backward sends
  fall from 24% to 14% of all sends, units sent straight back from 55% to 34%
  of own-base sends.
- **The rest is what the networks prefer:** both networks, playing their
  favourite, send about one in five own-base sends backward and send a third
  straight back. No entropy setting removes that; it has to be learned.
- `free_scratch`, already more decided, changes little (82% and 81%).
- Ratings by chance understate the large-bonus networks against other
  networks: the same network is 66.5% against itself by chance.

**Neural AI now plays at temperature 0.8** (the user's choice, 2026-10-10):
often its favourite, but not always. Built: `choose(..., temperature=)` in
`ml/models/policy.py` (the route's odds, then the amount's, raised to the
power 1 / temperature; for free amounts `amounts.sample_sharpened`), passed
on by `NeuralAgent(temperature=)`; `ais/neural_ai.py` sets `TEMPERATURE =
0.8`. Training and every measurement still draw at 1. Tests for both designs
(draws follow the sharpened odds, 1 changes nothing) and for the agent and
Neural AI; all 362 tests pass.

Measured the same way (`greedy_shuffling_script.py --temperature 0.8`, output
`temperature_0.8_shuffling.txt`; "plays its favourite" counts the route of every
move drawn):

| Neural AI's network | score | median game | backward, of all sends | sent back within 5 s | plays its favourite |
|---|---|---|---|---|---|
| by chance (1.0), vs strategic_ai | 88.5% | 81 s | 24% | 55% | about 28% |
| at 0.8, vs strategic_ai | 89.0% | 73 s | 24% | 54% | 33% |
| favourite, vs strategic_ai | 88.5% | 49 s | 14% | 34% | 100% |
| at 0.8 against by chance | 63.0% (+2.6 SE) | 90 s | 26% / 27% | 65% / 67% | 42% / 38% |
| at 0.8 against favourite | 42.0% (−1.6 SE) | 77 s | 25% / 21% | 64% / 51% | 44% / 100% |

- 0.8 is stronger than drawing by chance (63%) and a little weaker than the
  favourite (42%), but it shuffles as much as drawing by chance.
- It plays its favourite only a little more often: the network's odds are
  spread over several routes of about the same weight, and 0.8 barely
  separates them. Over 16,456 positions of its games against strategic_ai,
  the favourite route's share by temperature: 1.0 28%, 0.8 33%, 0.6 40%,
  0.5 45%, 0.4 52%, 0.3 61%, 0.2 72%, 0.1 85%.

**Then 0.2** (the user's choice, the same day): about 70% its favourite.
`ais/neural_ai.py` now says `TEMPERATURE = 0.2`. Measured the same way
(`--temperature 0.2`, output `temperature_0.2_shuffling.txt`):

| Neural AI's network | score | median game | backward, of all sends | sent back within 5 s | plays its favourite |
|---|---|---|---|---|---|
| at 0.2, vs strategic_ai | 90.5% | 52 s | 16% | 38% | 66% |
| at 0.2 against by chance | 63.5% (+2.7 SE) | 78 s | 22% / 28% | 54% / 70% | 71% / 41% |
| at 0.2 against favourite | 51.0% (even) | 78 s | 22% / 22% | 53% / 51% | 74% / 100% |

- 0.2 is as strong as always playing the favourite (51% against it, 63.5%
  against drawing by chance) and shuffles nearly as little (16% backward
  against 14% for the favourite and 24% by chance), while still varying its
  play a third of the time.

**The game runner draws at a temperature too** (for the home screen's
ratings): `play_games(..., temperatures={name: T})`, the same through
`RolloutPool.play` and its inference server (each network its own; others
at 1). Only for playing: the recorded probabilities stay the network's own,
so training leaves them at 1. `ml bot-ratings` plays Neural AI at
`ais.neural_ai.TEMPERATURE`. The matches recorded earlier against Neural AI
(Beast, the look-ahead, the overnight network) were played by chance, so they
now count for a separate player, `neural_ai drawing by chance` (the same
network at 1), which plays the quick players as well. Tests: each network
draws at its own temperature, temperatures must be above 0, the server draws
by the sharpened odds, and bot-ratings rates both.

New home-screen ratings (`ais/ratings.json`, 2026-10-10): Neural AI 1420
(1332 before, drawing by chance; that player now rates 1334), Beast 1477
(1473), Atlas 1268, Atlas Clean 1142, careful_ai 787, aggressive_ai 670.
Beast is still the strongest, by 57 points; its own rating rests on 120
recorded games.

**Beast against the new Neural AI** (the user's question, 2026-10-10: does
Beast play at 0.2 too?). No: Beast never draws by chance. Its candidates are
the network's 8 most likely sends (a temperature keeps their order) and it
plays the move its mix uses most, so a temperature changes nothing. Measured
directly instead (`greedy_shuffling_script.py --beast 0.2`, output
`beast_shuffling.txt`; 100 games each, maps 960,000–960,049, both seats;
about 40 s per game, 16 minutes in all):

| | score | median game | backward, of all sends | sent back within 5 s |
|---|---|---|---|---|
| Beast vs strategic_ai | 90.5% | 62 s | 12% | 32% |
| Beast against Neural AI at 0.2 | 53.0% (+0.6 SE) | 70 s | 17% | 38% |
| … Neural AI at 0.2 | 47.0% | | 22% | 52% |

- Beast and the new Neural AI are about even: 53 to 47 is within the noise.
  The home screen's 57 points between them (1477 against 1420) would mean
  about 58%; Beast's rating there comes only from older matches against
  Neural AI drawing by chance and against Atlas.
- Against strategic_ai both score 90.5%.
- Beast shuffles about as little as its network playing its favourite (12%
  backward, against 14%): its candidates come from that network, so it shares
  its preferences.
- Beast stays Beast: nothing has beaten it, but its lead is no longer clear.

## A longer lookahead from scratch (2026-10-11)

- [x] `free_lookahead` (`ml/scripts/free_lookahead.sh`, the user's plan;
      prepared 2026-10-10, run 2026-10-11 02:10 to 06:54): `free_scratch` again with `--lookahead 0.995`
      instead of 0.99, from the same start (`free_v2_seed0.pt`, made when
      `free_scratch` began), seed 0 so the same training maps, and the same
      small entropy bonus (0.001 halving every 200 rounds towards 0.0002).
      Checked by building the settings from the script's flags as `ml train`
      does: only `lookahead` differs (1 of 35). 300 rounds, then `ml compare`
      at rounds 100, 200 and 300 of both, and both rounds 300 head to head;
      `free_scratch` is the control and needs no rerun.
      - Why: playing their favourite, the networks still send one in five
        sends between own bases backward and a third straight back
        ("Playing the favourite instead of drawing by chance"); a shuffle
        only costs something seconds later, and at 0.99 half a move's credit
        is gone after 6.3 s (11.5 s at 0.995; advantages 1.5 times as
        spread, measured on `free_scratch` round 300).
      - No entropy split: at this small bonus the amounts' pull on the route
        is about 0.0005 per decision ("The entropy in its two parts").
      - The comparison's hours will favour `free_lookahead`: `free_scratch`'s
        7.0 hours include the memory problem and the swapping.
      - The run: 300 rounds in 4.7 hours (about 55 s per round, memory at
        0.8–2.1 GB, no problems), checked every hour.
      - **Results** (`ml compare --name free_lookahead`,
        `ml/runs/compare_free_lookahead/report.txt`; 100 games per match,
        strategic_ai fixed at 1000; one seed each):

        | round | lookahead 0.995 | lookahead 0.99 (`free_scratch`) |
        |---|---|---|
        | 100 | 1006, 48% vs strategic_ai | 1223, 84% |
        | 200 | 1121, 61% | 1245, 82% |
        | 300 | 1218, 78% | 1316, 88% |

        Rounds 300 head to head: 0.995 scored 35.5% of 100 games (−2.9
        standard errors from even). The longer lookahead was behind at every
        round; it learned more slowly (in the checks every 10 rounds, 34%
        against 71% vs strategic_ai over rounds 80–140) and had nearly
        caught up by rounds 210–270 (72% against 75%), but not quite.
      - It did not shuffle less. Averages over rounds 251–300, 0.995 against
        0.99: units away from the front 36% against 35%, sends to own bases
        75% against 76%. What changed instead: it waits more where it may
        send (26% against 10%, rising through the run from 13%), its
        self-play games are longer (85 against 78 s) and less often decided
        (85% against 92%), and it stayed less decided (entropy 2.45 against
        1.97 over rounds 51–150, 1.60 against 1.54 by the end).
      - Reading: with credit spread over 11.5 s instead of 6.3 s, each move's
        advantage carries more of what other moves did (more noise), so it
        learns more slowly, as expected from GAE; the extra reach did not
        buy a cheaper fix for shuffling. One seed only, so the size of the
        gap is uncertain, but its direction held at every round and in the
        head to head. 0.99 stays. Not a new best: Beast unchanged.

## Training free_scratch until it stops improving (from 2026-10-11)

**The user's plan (2026-10-11), four steps:**
1. `free_scratch` trains on until it stops improving.
2. Then the send limit, as in "Acting once a second": gap 10 until its
   rating with the limit lifted stops rising (about 200 rounds last time,
   where most of the gain came from), then gaps 9, 8, ... and down to 0
   (last time the ramp stopped at 5). The plateau of step 1 is its control.
3. Then a longer lookahead on the trained network: 0.995 for 100 rounds
   beside a 0.99 control from the same start, with shuffling measured
   directly (`ml/runs/logs/greedy_shuffling_script.py`). The user's view:
   its potential is larger once the network is trained. For: a trained
   network draws less at random, so a longer lookahead adds less noise, and
   `free_lookahead`'s gap was largest early. Against: it did not shuffle
   less at all there, and a better value estimate usually makes a longer
   lookahead matter less.
4. Then a judge for it. Beast's look-ahead cannot use free networks yet
   (it needs candidate amounts per route), so that is built first; then
   the new network proposing moves with `judge1` against Beast, which shows
   whether the judge is what limits it (before, better proposers reached
   only 51–52.5% against Beast); then a new judge from its games. judge3
   ("A third judge") stopped on 2026-10-04 after about 3,000 of its 37,500
   games and was never trained; this is that experiment with a stronger
   player.
If the network beats Beast at any point, it becomes Beast then.

- [x] Step 1 (`ml/scripts/free_scratch_plateau.sh`, 2026-10-11 10:59 to
      18:38): the same run carried on exactly where it stopped at round
      300 (weights, optimizer, step size; the entropy bonus is 0.00048 at
      round 300, and its part above the floor 0.0002 halves every 200
      rounds: 0.00034 at round 500, 0.00027 at 700), in blocks
      of 100 rounds. The 16-game checks during training cannot show a
      plateau (it beats strategic_ai about 88% of the time), so after each
      block its last round plays 200 games each, on the comparison's fresh
      maps and both seats, against the round 100 earlier, Neural AI's network
      (`deep_from_imitation` round 500, a fixed yardstick; both draw by
      chance) and strategic_ai (`ml compare --name free_scratch_plateau_N`).
      - The rule, set before any result: a block is flat when its last
        round scores under 55% against the round 100 earlier (the "no
        difference" line used for one seed); two flat blocks in a row end
        the run. Round 1500 ends it in any case (about 18 hours).
      - About 1.5 hours per block, then about 2 minutes of matches.
        Checked before starting: a 4-game dry run of the matches, the score
        read back from `results.json`, and the rule on 0.50/0.55/0.60.
      - Blocks (200 games per match, fresh maps, both seats; "SE": standard
        errors from even):

        | block's last round | against the round 100 earlier | against Neural AI's network | against strategic_ai | flat? | finished |
        |---|---|---|---|---|---|
        | 400 | 57.3% (+2.1 SE) | 42.8% (−2.1 SE) | 84.3% | no | 12:25 |
        | 500 | 47.0% (−0.8 SE) | 39.3% (−3.0 SE) | 82.5% | yes (1st) | 13:54 |
        | 600 | 58.3% (+2.3 SE) | 40.0% (−2.8 SE) | 86.5% | no (count back to 0) | 15:29 |
        | 700 | 47.3% (−0.8 SE) | 42.2% (−2.2 SE) | 84.3% | yes (1st) | 17:04 |
        | 800 | 47.8% (−0.6 SE) | 44.8% (−1.5 SE) | 88.0% | yes (2nd): the end | 18:38 |

      - **It stopped improving.** By the rule, flat from round 600 (two flat
        blocks, 700 and 800). The fixed yardstick says the same from earlier:
        against Neural AI's network 42.8%, 39.3%, 40.0%, 42.2%, 44.8% at rounds
        400 to 800, 41.8% over all 1,000 games, with no clear trend. The
        blocks against the round before alternated (57%, 47%, 58%, 47%, 48%),
        so part of what it learned between blocks beat its own earlier
        rounds without playing better in general.
      - So `free_scratch` levels off below Neural AI's network (both drawing
        by chance), about where the old design's 64×6 runs levelled off
        too. Not a new best: Beast unchanged.
      - In training (averages over the last 20 rounds of each block, rounds
        300 → 800): entropy 1.50 → 1.16, self-play games longer (78 → 96 s)
        and less often decided (92% → 87%), waits when it may send 10% →
        18%, sends per game 613 → 728, units away from the front unchanged
        (35%). 800 rounds took 14.4 hours of training in all.
      - The start for step 2: round 800 (the last; rounds 600–800 are level
        with each other and against the yardstick).
- [x] Step 2 (`ml/scripts/free_gap_ramp.sh`, the user's go 2026-10-11, started
      19:39): the send limit on `free_scratch` round 800, which is also the
      control. Every run keeps `free_scratch`'s settings except what a run
      from a trained network needs (checked by building the settings from
      the flags: 7 of 35 differ, all intended): no value-only rounds, no
      material hint, the step size where it stopped (3.0e-05), the entropy
      bonus carrying on its schedule (0.00025), the gap, the round count
      and the seed (training maps). The optimizer's memory starts afresh.
      - Phase 1, gap 10 (`free_gap10`, seed 10), in blocks of 50 rounds.
        After each, its last round with the limit lifted plays 200 games
        each against the round 50 earlier (the control for the first
        block), the control, Neural AI's network and strategic_ai. The rule,
        set before any result: a block is flat under 55% against the round
        50 earlier; two flat in a row end the phase, round 300 at the
        latest. (Last time, played unlimited, gap 10 beat the control 61%
        after 50 rounds and its rating peaked by round 200.)
      - Phase 2, from phase 1's last round: 50 rounds each at gaps 9, 8, 7,
        6, 5, 4, 3, 2, then 50 with no limit (gap 1 is no limit), each its
        own run (`free_gap9` ... `free_gap2`, `free_gap0`; seed = gap, 1 for
        no limit). After each, its last round with the limit lifted plays
        phase 1's last round, the control, Neural AI's network and
        strategic_ai, 200 games each (`compare_free_ramp_gapN`).
      - Checked before starting: one training round of the free network at
        gap 10 (12 games: 150 sends per game and seat, against about 700
        unlimited) and its matches with the limit lifted (4 games each, no
        gaps recorded), then removed.
      - Expected: about 6 to 10 hours.
      - Phase 1 blocks (limit lifted for the matches; 200 games each, fresh
        maps, both seats):

        | `free_gap10` round | against the round 50 earlier | against the control (`free_scratch` r800) | against Neural AI's network | against strategic_ai | flat? | finished |
        |---|---|---|---|---|---|---|
        | 50 | = the control | 58.8% (+2.5 SE) | 48.5% (−0.4 SE) | 88.5% | no | 20:09 |
        | 100 | 54.8% (+1.3 SE) | 64.5% (+4.1 SE) | 52.0% (+0.6 SE) | 87.5% | yes (1st; just under 55%) | 20:38 |
        | 150 | 57.8% (+2.2 SE) | 67.5% (+4.9 SE) | 56.2% (+1.8 SE) | 89.5% | no (count back to 0) | 21:06 |
        | 200 | 47.2% (−0.8 SE) | 66.0% (+4.5 SE) | 54.5% (+1.3 SE) | 88.0% | yes (1st) | 21:34 |
        | 250 | 47.8% (−0.6 SE) | 65.8% (+4.5 SE) | 55.5% (+1.6 SE) | 87.2% | yes (2nd: phase 1 ends) | 22:01 |

        Phase 1 ended at round 250 (flat from round 150). Its step size had
        come down from 3.0e-05 to 2.0e-05 by then.
      - Phase 2 steps (50 rounds each from the step before; limit lifted for
        the matches; 200 games each, fresh maps, both seats):

        | run (gap) | against the ramp's start (`free_gap10` r250) | against the control | against Neural AI's network | against strategic_ai | finished |
        |---|---|---|---|---|---|
        | `free_gap9` (9) | 52.2% (+0.6 SE) | 68.0% (+5.1 SE) | 64.2% (+4.0 SE) | 88.0% | 22:30 |
        | `free_gap8` (8) | 54.2% (+1.2 SE) | 69.5% (+5.5 SE) | 60.2% (+2.9 SE) | 89.5% | 22:58 |
        | `free_gap7` (7) | 56.2% (+1.8 SE) | 67.5% (+4.9 SE) | 60.0% (+2.8 SE) | 89.5% | 23:28 |
        | `free_gap6` (6) | 54.8% (+1.3 SE) | 70.0% (+5.7 SE) | 58.2% (+2.3 SE) | 90.5% | 23:56 |
        | `free_gap5` (5) | 57.8% (+2.2 SE) | 69.5% (+5.5 SE) | 61.0% (+3.1 SE) | 90.8% | 00:27 |
        | `free_gap4` (4) | 55.8% (+1.6 SE) | 72.8% (+6.4 SE) | 61.8% (+3.3 SE) | 90.0% | 00:59 |
        | `free_gap3` (3) | 57.5% (+2.1 SE) | 68.5% (+5.2 SE) | 64.5% (+4.1 SE) | 89.8% | 01:35 |
        | `free_gap2` (2) | 56.2% (+1.8 SE) | 69.5% (+5.5 SE) | 63.2% (+3.7 SE) | 89.5% | 02:13 |
        | `free_gap0` (none) | 57.5% (+2.1 SE) | 72.8% (+6.4 SE) | 60.8% (+3.0 SE) | 90.0% | 03:00 |

      - Reading: the send limit got this network past its plateau again, as
        it did the old design. Gap 10 alone gave most of it (the control
        58.8% → 67.5% by round 150; Neural AI's network 48.5% → 56%). The
        ramp then added a little: against the ramp's start every step scored
        52–58%, against the control 67.5–72.8%, against Neural AI's network
        58–64.5%, with no clear trend from step to step (each figure ±3.5
        points). Rounds got slower as the limit came off: 32 s at gap 8,
        43 s at gap 2, 52 s with none (more sends and decisions per game).
        The share of undecided choices rose from about 0.67 (gap 8) to 0.90
        (none). 7.4 hours in all (19:39–03:00).
- [x] Step 3, changed by the user (2026-10-11 23:15): when the ramp has
      finished (reached no limit), evaluate its result, then straight away
      train the strongest network without a send limit at lookahead 0.995
      for 150 rounds, "just to see what happens"; one arm, no 0.99 control.
      Then a break before the judge (step 4). Both start by themselves
      (`ml/scripts/free_after_ramp.sh`, started 23:20 under nohup; it waits
      for the ramp's last line; log `ml/runs/logs/free_after_ramp_log.txt`).
      Rules set before any result:
      - Yardstick: every network of step 2 (`free_gap10` every 50 rounds,
        `free_gap9` ... `free_gap0` round 50) has played the control and
        Neural AI's network on the same maps; its mean score over the two.
      - The strongest: the best by the yardstick other than the ramp's end
        (`free_gap0` r50) plays the ramp's end head to head, 400 games, limit
        lifted (`compare_free_ramp_best`). The ramp's end is the strongest if
        it scores at least 50%, else the other one.
      - Beast: the strongest plays Beast and strategic_ai, 200 games each,
        limit lifted, the network drawing by chance as in every comparison
        (`compare_free_beast_check`). At 57.5% or more (+2.1 SE) it becomes
        Beast (`ml beast set`; as Beast it plays exactly so).
      - The arm `free_lookahead_trained`: from the strongest, no send limit,
        lookahead 0.995, 150 rounds, training maps from seed 12, the step size
        where the strongest stopped, the entropy bonus carrying on its
        schedule, no value-only rounds, no material hint. Rounds 50, 100 and
        150 play the strongest (its start), the control, Neural AI's network
        and strategic_ai, 200 games each (`compare_free_lookahead_trained`).
        Its round with the best score against its start plays Beast and
        strategic_ai, 200 games each (`compare_free_lookahead_trained_beast`);
        the same 57.5% rule.
      - Without a 0.99 control, a gain over the start is not the lookahead's
        alone; the ramp's own steps (about 52–54% against the round before per
        50 rounds) are the rough comparison.
      - Fixed for this: `ml compare` loaded a panel player whose name ends in
        `.pt` as a checkpoint, and Beast's name does
        (`...?judge=ml/runs/judge1/best.pt`). Now a `lookahead:`/`value:`
        player is played as itself (new test in `ml/tests/test_compare.py`).
      - Expected: about 2.5 hours after the ramp ends (about 02:15). In the
        event the ramp ended at 03:00 and rounds without a limit take about
        52 s, so the arm takes about 2 hours 10 minutes.
      - Evaluation (03:00–03:26). The yardstick (mean against the control and
        Neural AI's network): `free_gap10` 0.536, 0.583, 0.619, 0.603, 0.606
        (rounds 50–250); gaps 9 to none 0.661, 0.649, 0.637, 0.641, 0.652,
        0.673, 0.665, 0.664, 0.667. Best other than the ramp's end: gap 4
        (0.673). Head to head, 400 games: the ramp's end 50.1% against it, so
        the strongest is the ramp's end, `free_gap0` round 50.
      - Against Beast, 200 games, limit lifted: **48.8% (−0.4 SE)**; against
        strategic_ai 90.8% (Beast's own: 90.5%). The network alone, drawing by
        chance and thinking nothing ahead, is as strong as Beast's look-ahead
        (built on Neural AI's network with `judge1`). Under 57.5%: Beast
        unchanged.
      - The arm `free_lookahead_trained` (03:26–05:44, 52 s per round; matches
        to 06:06). Limit lifted, 200 games each, fresh maps, both seats:

        | round | against its start (`free_gap0` r50) | against the control | against Neural AI's network | against strategic_ai |
        |---|---|---|---|---|
        | start (0.99) | | 72.8% | 60.8% | 90.0% |
        | 50 | 48.5% (−0.4 SE) | 71.5% (+6.1 SE) | 63.0% (+3.7 SE) | 89.5% |
        | 100 | 51.8% (+0.5 SE) | 72.8% (+6.4 SE) | 62.0% (+3.4 SE) | 88.2% |
        | 150 | 51.2% (+0.4 SE) | 72.2% (+6.3 SE) | 58.8% (+2.5 SE) | 87.8% |

        Round 100 (best against its start) against Beast: 51.5% (+0.4 SE),
        strategic_ai 88.2%. Under 57.5%: Beast unchanged.

        Training behaviour, means per block of 50 rounds (self-play), against
        the 0.99 rounds before it without a limit (`free_gap0`):

        | | waits when it may send | units away from the front | game length | decided | entropy route / amounts | sends per game and seat | value error |
        |---|---|---|---|---|---|---|---|
        | `free_gap0` 1–50 (0.99) | 18.5% | 32.4% | 93 s | 85.3% | 0.60 / 0.28 | 679 | 0.103 |
        | arm 1–50 (0.995) | 19.6% | 31.5% | 89 s | 86.3% | 0.58 / 0.26 | 642 | 0.105 |
        | arm 51–100 | 20.4% | 31.8% | 87 s | 87.4% | 0.58 / 0.26 | 623 | 0.108 |
        | arm 101–150 | 20.6% | 33.2% | 90 s | 86.5% | 0.61 / 0.28 | 645 | 0.108 |

      - Reading: 150 rounds at 0.995 changed nothing measurable. Against its
        start it stayed at 48.5–51.8%, against the control and Beast where the
        start was; against Neural AI's network it drifted from 63% to 59%
        (within the noise of two 200-game figures). Its play barely moved:
        slightly more waiting (18.5% → 20.6%, against 10% → 26% when 0.995
        trained from scratch), games a few seconds shorter, a few percent
        more decided, the value error a little higher (a longer lookahead's
        targets are noisier). So the longer lookahead did not hurt a trained
        network as it hurt one from scratch, but it did not help either. The
        user's view (more potential once trained) is not borne out within 150
        rounds. Without a 0.99 control it stays open whether the network had
        simply stopped improving without a limit; the ramp's steps had already
        levelled off (52–58% against its start, no trend).
      - Then a break (the user's plan); next, step 4, the judge.
- [x] The strongest at temperature 0.2 against Beast (the user's request,
      2026-10-12; `ml/scripts/beast_match.py`, new: one network against Beast
      with the score live, game by game). `free_gap0` round 50 drawing at 0.2
      (as Neural AI plays), no send limit, 200 games on the comparisons' maps,
      both seats, the same maps as `compare_free_beast_check`; 21 minutes.
      Result in `ml/runs/logs/beast_match_free_gap0_r50_t0.2_maps300.json`.

      | temperature | against Beast | W / D / L | as red | as blue |
      |---|---|---|---|---|
      | 1 (by chance; `compare_free_beast_check`) | 48.8% (−0.4 SE) | | | |
      | 0.2 | **56.0% (+1.7 SE)** | 107 / 10 / 83 | 63.5% | 48.5% |

      - Sharper choices gain about 7 points against Beast (the difference of
        two 200-game scores: about ±5 points, so +1.4 SE).
      - Just under the 57.5% bar: Beast stays. Per map: both games won on 30
        maps, split on 50, both lost on 20.
      - The seat split (63.5% as red, 48.5% as blue, ±5 each) was not looked
        for in advance; it may be noise.
      - Before it could become Beast: Beast plays a checkpoint by
        `load_player`, which draws at temperature 1; its name would need to
        carry the temperature.
      - Next (agreed 2026-10-12, set before playing it): a second batch of
        200 games on the next 100 maps (`--first-map 400`, maps
        900400–900499); over all 400 games, +2.0 SE or more (55.0%) makes it
        Beast. The script now adds up all batches of the same network,
        temperature and Beast, with that rule.
      - Second batch (maps 900400–900499, 19 minutes): 55.0% (+1.4 SE),
        106 / 8 / 86; as red 58.5%, as blue 51.5%.
      - **All 400 games: 55.5% (+2.2 SE), 213 / 18 / 169. Clearly ahead by
        the rule: `free_gap0` round 50 at temperature 0.2 is the new Beast**
        (`ml beast set "network:ml/runs/free_gap0/round_0050.pt?temperature=0.2"`,
        2026-10-12). The first time a network on its own, thinking nothing
        ahead, is our strongest player since the look-ahead with `judge1`
        took over on 2026-10-02. It is also far quicker: about 4 ms a tick
        in the game window.
      - The seat split held in the second batch: 61% as red, 50% as blue
        over all 400 (±3.5 each). Every map is played from both seats, so the
        score is fair either way, but the network seems better at one side's
        starts. Not looked into yet.
      - New for this: a player name `network:PATH?temperature=T`
        (`ml/play/agent.py`, `ml/play/players.py`): a network on its own at a
        temperature, wherever players are named (the evaluator, the
        rollouts, Beast). Beast had no way to play a checkpoint other than
        by chance. Test in `ml/tests/test_agent.py`.
      - `ml bot-ratings`: Beast is quick now, so it plays the quick players
        directly (it was rated from recorded matches); the old Beast's
        recorded matches (against Atlas and Neural AI's network by chance)
        stay, as `OLD_BEAST`, with the new Beast's 400 games against it.
      - New home-screen ratings (`ais/ratings.json`, 2026-10-12; 2 minutes):
        Beast 1440 (900 games), Neural AI 1433, Atlas 1242, Atlas Clean 1132,
        careful_ai 783, aggressive_ai 681; the old Beast 1413 (1477 before),
        Neural AI's network by chance 1323. The new Beast's own matches here,
        100 games each: strategic_ai 93.5%, careful_ai and aggressive_ai 98%,
        Neural AI's network by chance 62.5%, **Neural AI (at 0.2) 48.5%**.
      - So Beast is top only by 7 points: Neural AI at 0.2 is about as strong
        (the old Beast scored 53% against it, the new one 48.5%; ±5 each).
        Sharper choices helped Neural AI's network much more (by chance 1323,
        at 0.2 1433) than the free network (against Neural AI's network by
        chance: 60.8% by chance, 62.5% at 0.2). To settle which of the two is
        stronger: Neural AI's network at 0.2 against the new Beast, e.g.
        `beast_match.py --network ml/runs/deep_from_imitation/round_0500.pt`.

## A judge for the free network (step 4, from 2026-10-12)

**The aim:** a look-ahead built on `free_gap0` round 50 that clearly beats
the new Beast (that network alone at 0.2). Last time this route gave a lot:
on Neural AI's network, the look-ahead with `judge1` beat the same
look-ahead judged by the network itself 59.5%, and the plain network 65%.

**What stood in the way:** the look-ahead could not use a free network (it
ranks "the 8 likeliest sends", and a free network has no fixed amounts to
rank), and the judge model reads only the old inputs (14 base features; the
free network uses v2, 34).

**The plan, agreed with the user (2026-10-12), a pause after each stage:**
1. The look-ahead for free networks.
2. Does a look-ahead help this network at all? The look-ahead judged by the
   network's own value output, and by `judge1`, each against Beast, 200
   games. Informs, does not stop the run.
3. Judge games `round4`: about 20,000 games, 8 positions per seat; `free_gap0`
   mostly against itself (about 85%), at 0.2 for most games and at 1 for
   about a third (so the judge also sees the less likely positions the
   look-ahead considers for the opponent); also Neural AI's network at 0.2,
   a little strategic_ai and careful_ai; not the old look-ahead Beast (about
   40 s a game). Needs `judge-games` to play a network at a temperature and
   the judge model to read v2 inputs.
4. `judge4`, twice: from `free_gap0`'s layers and from random weights; the
   better one goes on.
5. The look-ahead with `judge4` against Beast (200 games, 200 more if close;
   +2 SE over all to replace it), against the look-ahead with `judge1` or
   its own value output, and against Atlas.

- [x] Stage 1 (2026-10-12): the look-ahead proposes for a free network its
      most likely routes, each at its most likely amount
      (`_free_favourites` in `ml/play/value_player.py`). Single amounts are
      not ranked across routes as the option network's are: a free network
      spreads a route's probability over every amount, so a likely route
      with a wide spread would lose to a narrow one. A new setting
      `amounts=N` (`lookahead:PATH?amounts=2`) also tries the amounts the
      route's other curves aim at, heaviest first; the sends still number
      `mine` and `theirs` in all, so fewer routes fit. Tests in
      `ml/tests/test_lookahead.py`.
      Demo (one position at tick 150, and one game against strategic_ai
      each, map 990,007):
      - Judged by the network's own value output, the table is nearly flat:
        almost every cell 0.29–0.32, hardly changed by the opponent's moves.
        `judge1` spreads the same kind of table from −0.53 to +0.83 and
        reacts strongly to the opponent's moves.
      - Both won their game; about 40–44 ms a tick (63–72 positions judged
        a tick), as the old Beast.
      - `ml/scripts/beast_match.py` now also plays any named player
        (`--player NAME --label LABEL`); stage 2 is
        `sh ml/scripts/judge4_stage2.sh` (two runs of it, about 25 minutes
        each, on maps 900300–900399 like Beast's own deciding match).
- [x] Stage 2 (2026-10-12, `sh ml/scripts/judge4_stage2.sh`, run by the user;
      200 games each against Beast, maps 900300–900399, both seats):

      | look-ahead on `free_gap0` r50, judged by | against Beast | W / D / L | as red | as blue |
      |---|---|---|---|---|
      | its own value output | 34.5% (−4.4 SE) | 61 / 16 / 123 | 41.5% | 27.5% |
      | `judge1` | 32.8% (−4.9 SE) | 54 / 23 / 123 | 40.5% | 25.0% |

      - **The look-ahead makes this network much weaker, whichever judge.**
        The two are level (1.7 points apart, ±4.7 on the difference); the
        expectation that `judge1` would do clearly better was wrong.
      - Why, most likely: the network sends about 7 times a second, often
        small exact amounts, as a flow over many ticks; the look-ahead judges
        one send per side at one tick, which barely changes the position, so
        its own value output picks by noise (0.29–0.32 across the table in
        the demo) and `judge1`, which learned the old network's games and
        inputs, confidently misjudges this player's positions. And the old
        look-ahead's gain was mostly against drawing by chance (65% against
        the plain network by chance, 47–53% against Neural AI at 0.2):
        temperature 0.2 already takes most of it.
      - So a `judge4` from this network's games is unlikely to pay: even the
        network's own judgement starts at 34.5%, and the last dedicated judge
        gained about 9 points over that head to head; about 20 would be
        needed to break even. The judge as critic in training failed before
        (night of 2026-10-03). Recommended to the user: park the judge route;
        cheap checks instead (Beast's temperature; Beast against Neural AI),
        and training (a second seed, a larger network, a pool of past
        opponents). Their call.
- **Parked** (the user, 2026-10-12): no judge games or `judge4` for now. The
  look-ahead for free networks stays in the code.

## A second round of the send-limit ramp (from 2026-10-12)

**The user's request (2026-10-12):** park the judge; train the model again
from gap 10 down to no limit, a second round of the ramp; then the best
network of this round and its final one against `free_gap0` round 50 (today's
Beast) and strategic_ai, 200 games per pairing, 1,200 in all.

**The design** (`ml/scripts/free_gap_ramp2.sh`, started 2026-10-12 23:35
under nohup, log `ml/runs/logs/free_gap_ramp2_log.txt`): the first round's,
from `free_gap0` round 50, which is also the control.
- Phase 1: gap 10 (`free2_gap10`, seed 210), blocks of 50 rounds; after
  each, its last round with the limit lifted plays the round 50 earlier,
  the control, Neural AI's network and strategic_ai, 200 games each (by
  chance, as every comparison; `compare_free2_gap10_N`). Flat under 55%
  against the round 50 earlier; two flat blocks end the phase, round 300
  at the latest.
- Phase 2: gaps 9 to 2 and none, 50 rounds each (`free2_gap9` ... `free2_gap0`,
  seeds 209 ... 202, 201), each step played against phase 1's last round,
  the control, Neural AI's network and strategic_ai (`compare_free2_ramp_gapN`).
- The step size where the first round stopped (2.0e-05), the entropy bonus
  about at its floor (0.000204), no value-only rounds, no material hint.
- Evaluation, set before any result: "the best" is the first round's
  yardstick (mean score against the control and Neural AI's network, limit
  lifted, by chance) over every network of this round except the final one
  (`free2_gap0` round 50). The best, the final one and `free_gap0` round 50
  play each other and strategic_ai, 200 games per pairing, at temperature
  0.2 as Beast plays, no send limit (`ml/scripts/round_robin.py`, new: every
  pairing, scores live, ratings fitted with strategic_ai at 1000; output
  `ml/runs/logs/round_robin_free2.json`). Temperature 0.2 because that is
  how Beast plays and where the first round's edge over Neural AI's network
  disappeared; the steps' comparisons give the by-chance view.
- Beast's rule: a new network at +2.0 SE or more (57.1% of 200) against
  `free_gap0` round 50 at 0.2 becomes Beast (if both do, the winner of
  their head to head).
- Expected: about 8 hours.
- Phase 1 blocks (limit lifted for the matches; 200 games each, fresh maps,
  both seats):

  | `free2_gap10` round | against the round 50 earlier | against the control (`free_gap0` r50) | against Neural AI's network | against strategic_ai | flat? | finished |
  |---|---|---|---|---|---|---|
  | 50 | = the control | 53.5% (+1.0 SE) | 68.0% (+5.1 SE) | 89.0% | yes (1st) | 00:05 |
  | 100 | 51.2% (+0.4 SE) | 52.8% (+0.8 SE) | 62.5% (+3.5 SE) | 90.0% | yes (2nd: phase 1 ends) | 00:32 |

  Phase 1 ended at round 100, flat from the start: gap 10 gave little this
  time (the first round: 58.8% → 67.5% against its control by round 150).
  Step size 2.0e-05 → 1.5e-05; about 30 s a round.
- Phase 2 steps (50 rounds each from the step before; limit lifted for the
  matches; 200 games each, fresh maps, both seats):

  | run (gap) | against the ramp's start (`free2_gap10` r100) | against the control | against Neural AI's network | against strategic_ai | finished |
  |---|---|---|---|---|---|
  | `free2_gap9` (9) | 52.0% (+0.6 SE) | 54.2% (+1.2 SE) | 61.8% (+3.3 SE) | 89.8% | 01:00 |
  | `free2_gap8` (8) | 50.8% (+0.2 SE) | 50.8% (+0.2 SE) | 63.0% (+3.7 SE) | 89.5% | 01:28 |
  | `free2_gap7` (7) | 47.2% (−0.8 SE) | 55.0% (+1.4 SE) | 66.2% (+4.6 SE) | 89.2% | 01:57 |
  | `free2_gap6` (6) | 47.2% (−0.8 SE) | 53.8% (+1.1 SE) | 62.8% (+3.6 SE) | 91.5% | 02:28 |
  | `free2_gap5` (5) | 54.8% (+1.3 SE) | 56.0% (+1.7 SE) | 63.5% (+3.8 SE) | 90.2% | 03:00 |
  | `free2_gap4` (4) | 53.5% (+1.0 SE) | 57.2% (+2.1 SE) | 65.0% (+4.2 SE) | 91.8% | 03:33 |
  | `free2_gap3` (3) | 56.2% (+1.8 SE) | 59.0% (+2.5 SE) | 65.0% (+4.2 SE) | 88.2% | 04:08 |
  | `free2_gap2` (2) | 47.8% (−0.6 SE) | 54.0% (+1.1 SE) | 69.2% (+5.4 SE) | 89.2% | 04:46 |
  | `free2_gap0` (none) | 56.5% (+1.8 SE) | 56.8% (+1.9 SE) | 64.5% (+4.1 SE) | 90.2% | 05:33 |

  Rounds took 30 s at gap 10 and 54 s with no limit. The steps scored 47–59%
  against the control, at most +2.5 SE (gap 3), with a slight rise towards
  the end (gaps 4, 3 and none: 57–59%), but each figure is ±3.5 points.
- The yardstick (mean score against the control and Neural AI's network,
  limit lifted, by chance; the script's table):

  | network | yardstick | against the control | against Neural AI's network |
  |---|---|---|---|
  | `free2_gap10` r50 | 0.608 | 0.535 | 0.680 |
  | `free2_gap10` r100 | 0.576 | 0.527 | 0.625 |
  | `free2_gap9` r50 | 0.580 | 0.542 | 0.618 |
  | `free2_gap8` r50 | 0.569 | 0.507 | 0.630 |
  | `free2_gap7` r50 | 0.606 | 0.550 | 0.662 |
  | `free2_gap6` r50 | 0.583 | 0.537 | 0.627 |
  | `free2_gap5` r50 | 0.598 | 0.560 | 0.635 |
  | `free2_gap4` r50 | 0.611 | 0.573 | 0.650 |
  | **`free2_gap3` r50 (the best)** | **0.620** | 0.590 | 0.650 |
  | `free2_gap2` r50 | 0.616 | 0.540 | 0.693 |
  | `free2_gap0` r50 (the final one) | 0.606 | 0.568 | 0.645 |

  The best other than the final one is `free2_gap3` round 50, though the top
  six are within 0.015 of each other.
- The round robin (`ml/runs/logs/round_robin_free2.json`, 05:33–05:36): at
  temperature 0.2, no send limit, maps 900300–900399 from both seats (the
  comparisons' maps), 200 games per pairing, 1,200 in all. The row's score:

  | | against `free_gap0` r50 (Beast) | against `free2_gap3` r50 (best) | against `free2_gap0` r50 (final) | against strategic_ai | rating |
  |---|---|---|---|---|---|
  | `free2_gap0` r50 (final) | **60.8% (+3.0 SE)**, 106-63, 31 draws | 49.3% (−0.2 SE) | | 89.8% (175-16, 9 draws) | **1400** |
  | `free2_gap3` r50 (best) | 56.5% (+1.8 SE), 95-69, 36 draws | | 50.7% (+0.2 SE), 85-82, 33 draws | 90.0% (176-16, 8 draws) | 1395 |
  | `free_gap0` r50 (Beast) | | 43.5% | 39.2% | 90.0% (176-16, 8 draws) | 1344 |

  Ratings fitted to these games alone, strategic_ai = 1000.
- **Beast's rule** (+2.0 SE, 57.1% of 200, against `free_gap0` r50 at 0.2):
  the final one clears it (60.8%), the best does not (56.5%, just short).
  **`free2_gap0` round 50 is Beast** from 2026-10-13
  (`network:ml/runs/free2_gap0/round_0050.pt?temperature=0.2`, `ml beast set`).
  In `ml/evaluation/bot_ratings.py` the previous Beast is now `PREVIOUS_BEAST`
  and its deciding match (121.5 of 200 for the new one) is recorded.
- Reading: the second round helped, less than the first. With moves drawn by
  chance (the steps' comparisons) the final network beat its start by 56.8%
  (+1.9 SE); the first round's end beat its control by 72.8%. At temperature
  0.2, where Beast plays, the gain is plainer: 60.8% (+3.0 SE). This is the
  opposite of what the first round showed against Neural AI's network, whose
  60.8% at temperature 1 shrank to 48.5% at 0.2: there the gain was in
  playing by chance, here it holds up when playing nearly at its best.
  Gap 10 alone, which gave the first round most of its gain, gave nothing
  measurable this time (flat from the first block). Against strategic_ai all
  three score 90%: it no longer tells them apart. The best by the yardstick
  and the final one are even head to head (50.7%). 6 hours in all
  (23:35–05:36).
- `ml bot-ratings` rerun (2026-10-13, `ais/ratings.json`; log
  `ml/runs/logs/bot_ratings_2026-10-13.txt`): Beast 1474 (was 1440),
  Neural AI 1423, Atlas 1241, Atlas Clean 1133, careful 790, aggressive 683;
  the previous Beast 1426, the look-ahead Beast 1403. The new Beast scored
  60.5% against Neural AI over 100 games (+2.1 SE; the previous one 48.5%):
  the first Beast clearly ahead of Neural AI at the temperature both play at.
