# What we tried, as a tree (as of 2026-10-04)

A map of every experiment, by the choice it explored rather than by date.
Each branch says how far we went and why we stopped. Details, numbers and
logs are in `history.md`, in the sections named on the right of each topic
("history: ..."). The code of
dead branches (distillation, the search player, teaching from the
look-ahead, the judge as critic) was removed on 2026-10-04; their results
stay in `history.md`.

Ratings: our scale, `strategic_ai` = 1000, from `ml compare` (100 games per
match, about ±15–30 per checkpoint). "h2h" = head to head.

```
✅ in use    ▶ running    ↑ ahead, not in use yet    ✗ tried, stopped (reason)    ≈ no difference    ○ not tried yet

THE GAME AI
│
├── 1. WHAT THE NETWORK SEES (inputs)                                   history: Stage 2, "Inputs", "Units"
│   ├── ✅ v1: per base (owner, units, production, fleets on their way, its future
│   │        once they land), per pair (route, travel and shortest-path time),
│   │        global (shares, time). Unit counts on a log scale.
│   │        └── the best network still reads only these (it predates the rest)
│   ├── ▶ v2 (the user's set, 2026-10-08): counts as plain numbers (/100), the forecast
│   │        at 5/10/20/50/80 ticks and after every fleet has landed, a front flag,
│   │        distance to the nearest enemy base; per route the capture cost and "forward".
│   │        Inputs built (stage 1); a network reads them by name with feature_set=v2 (stage 3);
│   │        trained only together with free amounts so far (see 2.: ahead at round 300),
│   │        so not judged on its own
│   │        (history: "A new feature set")
│   ├── ≈ + route table (cost of capturing along every route, payback time)
│   │        and production shares: copying strategic_ai 72/68% vs 74/66% without.
│   │        Kept (appended; new networks read them), not shown to help.
│   ├── ≈ unit counts as plain numbers instead of log (judge): same loss, same play
│   ├── ✗ stopped asking "what inputs are missing": taking the copy mistakes apart
│   │        showed the network already plays ~89% of strategic_ai's sends;
│   │        defence (3% of sends) is the real gap, and inputs did not change it
│   └── ○ neighbourhood totals, distance through own territory, three-player
│            inputs (two enemies told apart)
│
├── 2. WHAT IT CAN DO (actions)                                         history: Stage 3, "A larger network"
│   ├── ✅ wait, or a route plus an amount: "just enough" + eighths of the units
│   │        (eighths reproduce 93% of strategic_ai's sends; quarters only 54%)
│   ├── ▶ free amounts: any whole number of units, from a blend of curves per route
│   │        (built 2026-10-08, stage 2; every trainer handles it since 2026-10-09, stage 3);
│   │        no repeated options by construction
│   │   ├── ≈ first look, with v2 inputs, self-play from scratch, 100 rounds, one seed
│   │   │        (free_scratch): 1122 against 1108 for the old design, 76% vs 61.5%
│   │   │        against strategic_ai, but 44% head to head (−1.2 SE); far more decided
│   │   │        (entropy 1.9 vs 3.7, with a smaller bonus); 1.4× the time per round
│   │   ├── ↑ carried on to round 300: 1289 against 1193 for the old design at its
│   │   │        round 300, 86% vs 74% against strategic_ai, 63% h2h (+2.6 SE); about
│   │   │        1300 on the Stage 10 scale (deep_scratch round 500: 1263, Neural AI:
│   │   │        1357). One seed, smaller entropy bonus, 1.3× the time per round
│   │   ├── ✅ carried on until it stops improving (2026-10-11, the user's plan, step 1 of 4:
│   │   │        then the send-limit ramp, a longer lookahead with a control, a judge):
│   │   │        rose to round 400 (57% vs r300), flat from 600 (r700, r800: 47–48% vs
│   │   │        the round 100 earlier); about 42% vs Neural AI's network from r400 to r800:
│   │   │        levels off below it. Start of step 2: round 800
│   │   ├── ✅ step 2: one send a second on round 800 until it stops gaining (to round 250,
│   │   │        flat from 150), then gaps 9 ... 2 and none, 50 rounds each: the end
│   │   │        (free_gap0 r50) beats the control 72.8%, Neural AI's network 60.8%,
│   │   │        and plays Beast even (48.8%); the strongest network so far
│   │   ├── ≈ step 3: free_gap0 r50 without a limit at lookahead 0.995 for 150 rounds,
│   │   │        one arm (free_lookahead_trained, 2026-10-12): 48.5–51.8% against its
│   │   │        start, Beast 51.5%, play barely changed; neither helps nor hurts. 0.99 stays
│   │   ├── ✅ free_gap0 r50 at temperature 0.2 against the look-ahead Beast: 55.5% over
│   │   │        400 games (+2.2 SE): the new Beast (2026-10-12), a network on its own
│   │   ├── ✗ step 4: a judge for the free network. The look-ahead takes free networks
│   │   │        (likeliest routes at their likeliest amounts), but on free_gap0 it hurts:
│   │   │        34.5% (own value) and 32.8% (judge1) against Beast (2026-10-12). Parked
│   │   ├── ✅ a second round of the ramp from free_gap0 r50 (2026-10-13): gap 10 flat from
│   │   │        the start, steps 47–59% against the start by chance; at temperature 0.2 the
│   │   │        final one (free2_gap0 r50) beat free_gap0 60.8% (+3.0 SE): the new Beast
│   │   └── ▶ the same with the old entropy bonus (0.01 throughout), to tell the design
│   │            from the bonus (free_old_entropy, 2026-10-10; paused at round 124)
│   ├── ○ one option per different amount: with 1–3 units the eighths repeat (2 units:
│   │        1,1,1,1,2,2,2,2); half of unlimited sends are like that, and 40–60% of the
│   │        amount entropy is spread over duplicates (history: "Amount options that send...")
│   ├── ✗ + "hold" (take and keep through fleets on their way) and "surplus"
│   │        (everything the source can spare safely): offered in only 0.5% and
│   │        2.5% of sends, self-play did not take them up. Kept; harmless.
│   └── ablation: at most one send a second (it sends 7 a second)     history: "Acting once a second"
│       ├── ✗ limited, untrained: 980 against 1356 (19.5% h2h): speed is part of its strength
│       ├── ✅ 150 rounds under the limit (send_gap10): 1159 (+179, 77% vs untrained), still
│       │        27% vs the unlimited control; less shuffling (65% vs 78% to own bases),
│       │        firmer amounts, but no more holding back (waits 31%, as untrained)
│       ├── ✅ the same network with the limit lifted, no retraining: beats the unlimited
│       │        control 61% (r150, 400 games) and its round 300 60.5%, rising with limited
│       │        rounds; what carried over: firmer amounts, not fewer sends. One seed so far
│       ├── ✅ more rounds under the limit (151-300), rated with it lifted: 1469 → 1485 by round
│       │        200, then flat; about +80 over the control (1404), 64% against it
│       ├── ✅ lifting it step by step from round 200: gaps 9, 8, 7, 6, 5, 50 rounds each: the
│       │        gain held (68% vs the control at gap 5, 54% vs the start); shuffling creeps back
│       └── ○ then without it; a second seed
│
├── 3. THE NETWORK (architecture and size)                              history: "Network size", "A larger network"
│   ├── graph transformer: every base a token, attention biased by routes
│   │   ├── 64 wide × 3 layers (120k weights) ............ superseded: plateau ~1175–1195
│   │   ├── ✅ 64 × 6 (221k) ............................... +80–100 over 64×3 (61.5% h2h)
│   │   ├── ✗ wider 128×3, deeper route head, 128×4, 192×4 (copying strategic_ai):
│   │   │        70–72% of its sends, like 64×6 (71%); the largest overfit
│   │   ├── ✗ 128 × 8 (1.1M) + capture output, kickstarted from the best 64×6:
│   │   │        no gain in 150 self-play rounds (49.5% h2h). Both arms plateaued
│   │   │        the same way, so size was not the limit then; the entropy bonus was.
│   │   └── ▶ 256 × 8 (4.5M) from random weights, self-play only, entropy 0.01 shrinking
│   │            (half-life 100, floor 0.0005), 500 rounds on Colab's A100 (scratch_256),
│   │            against deep_scratch (64×6, the same route, entropy 0.01)
│   └── ○ other architectures (per-fleet tokens, arrival timelines as input)
│
├── 4. WHERE TRAINING STARTS                                            history: Stage 10
│   ├── random weights (deep_scratch, 64×6) ......... 1263 at round 500
│   │        ○ does it catch up later? one seed only
│   └── ✅ imitation of strategic_ai (20 epochs, 11 min) ... 1357 at round 500 (+95, 65.5% h2h)
│
├── 5. HOW IT LEARNS                                                     history: Stages 8–9, 11, "Tuning", the night sections
│   ├── ✅ PPO self-play (96 games a round, GAE, self-adjusting step size)
│   │   ├── settings
│   │   │   ├── ✗ entropy bonus 0.01 (rounds 0–900): fine early, then the amounts became
│   │   │   │        nearly random (95% of an even pick) and progress stopped (~+50 / 400 rounds)
│   │   │   ├── ✅ entropy bonus 0.001 (rounds 900–1050): 67% h2h vs plain, +~100, still rising
│   │   │   ├── ≈ shrinking bonus 0.001 → 0.0004 (rounds 1050–1350): flat (1404 → 1394);
│   │   │   │        the step from 0.01 to 0.001 was a one-off, the 64×6 plateaus again
│   │   │   ├── ≈ entropy 0.003 / 0 early on, larger steps per round (20-round sweeps)
│   │   │   └── ✗ higher step-size ceiling (0.003): clearly worse
│   │   ├── rules of thumb added to the learning signal ("potentials")
│   │   │   ├── ≈ material hint (fading over 50 rounds): no effect
│   │   │   ├── ≈ production share (permanent): no effect on strength or behaviour
│   │   │   └── ≈ distance of units to the front (permanent): no effect, units did not move forward
│   │   │            ○ much larger weights; or a direct cost for shuffling to own bases
│   │   ├── how far ahead a move is credited (lookahead, GAE's λ; 0.99 in use: half the
│   │   │   credit gone after 6.3 s)
│   │   │   └── ✗ 0.995 (11.5 s) from scratch, as free_scratch (free_lookahead, 2026-10-11):
│   │   │            behind at every round (r300 1218 vs 1316, h2h 35.5%), learned more slowly,
│   │   │            shuffled as much, waited more (history: "A longer lookahead from scratch")
│   │   ├── the critic (what a move is compared with)
│   │   │   ├── ✅ the network's own value output
│   │   │   ├── ✗ judge2, frozen: the network learned to exploit it, collapsed (0% vs start)
│   │   │   └── ✗ judge2, learning along: no collapse, but −130 and longer games
│   │   │            (it predicts who wins, not how soon)
│   │   └── extra outputs
│   │       └── ✗ capture prediction ("is the target ours once the fleet lands"):
│   │                learns well (error 0.13), no measurable help so far
│   ├── ≈ league (past champions + scripted AIs, promotion gate): same as plain self-play
│   ├── ✗ learning from play-time help
│   │   ├── ✗ distilling move scoring's best move: 1195 → 968, the network got worse
│   │   ├── ✗ distilling search: one step, inconclusive, a minute per game
│   │   └── ✗ soft targets from the look-ahead (teach1/teach2, moves not options):
│   │            no effect (49.8%); the judge's differences between candidates are noise
│   └── ○ more games per update; decaying entropy from a fresh start; three-player games
│
├── 6. THINKING DURING THE GAME (play-time help)                        history: Stage 11, "Move scoring", "Two-move look-ahead", "The judge"
│   ├── search: play candidates out 6 s ahead ....... superseded (30 ms a tick)
│   ├── move scoring: judge the position after each of 8 candidates ... superseded
│   │        (judging every move instead: much worse, it finds the judge's mistakes)
│   ├── ✅ two-move look-ahead: our 8 × their 8 candidates, nash = BEAST
│   │   ├── judged by the network's value output ...... 66.7% vs Atlas
│   │   ├── ✅ judged by judge1 (separate value network, 12,500 games) ... 82.5% vs Atlas
│   │   ├── ≈ judge2 (+37,500 games): 53.5% vs Beast
│   │   ├── ≈ better networks proposing the candidates (ab_plain 150, shaping_entropy 150):
│   │   │        51% and 52.5% vs Beast: the judge decides, not the proposer
│   │   └── ▶ judge3: trained on the stronger network's games (round3, 37,500 games), running
│   └── ✗ always playing the network's favourite (no thinking): +3 points, within the noise
│
└── 7. TOOLS AND SPEED                                                   history: "Measurements", "A larger network"
    ├── ✅ CPU workers each running the network: 11 workers, not 6 (playing 15 against 20 s a round)
    ├── ✗ one GPU server for all workers, on this Mac: correct, but no faster
    ├── ✅ the same on Colab's A100 (ml colab): playing 31–33 s a round at any size up to
    │        256x8; rounds 1.3x (64x6) to 2.7x (256x8) faster than the Mac
    └── ✅ evaluation: fresh maps, ml compare ratings, ml behaviour, ml lineage, Atlas as yardstick
```

## The main line: how the best network was made

```
random weights ─► imitation of strategic_ai ─► self-play, entropy 0.01 ─► self-play, entropy 0.01 ─► self-play, entropy 0.001 ─► ▶ shrinking entropy
  (64×6)          deep_imitation, 20 epochs     deep_from_imitation          ab_plain                     shaping_entropy              entropy_decay
                                                rounds 0–500: 969 → 1357     500–900: 1357 → 1405         900–1050: 1405 → 1487        1050–1350
```

`ml monitor --run lineage_shaping_entropy` shows all of it as one set of curves.

The best players today:
- **on instinct:** `shaping_entropy` round 150 (1487 linked rating; 68.5% against round 500)
- **with thinking (Beast):** the two-move look-ahead with `deep_from_imitation` round 500 and `judge1`
  (82.5% against Atlas; newer networks with judge2 reach 51–52.5% against it)

## Open branches, roughly by expected value

1. ▶ **Keep training with a small, shrinking entropy bonus** (running). The one
   thing that still moves the network.
2. ○ **A new judge from the stronger networks' games**, the way to a better Beast.
3. ○ **The larger network again**, now that self-play improves: its earlier
   test ran at the setting that stalled everything.
4. ≈ **Shuffling between own bases**: mostly not shuffling. About 80% of sends
   to own bases move units closer to the nearest enemy base (relays, since a
   send can only cross one route), two thirds land on a front base; about
   20% move away (2026-10-08, history: "Sends between own bases").
5. ○ **Random start vs imitation start, further**: does the random start catch up?
6. ○ **Three-player games**: not started; needs inputs that tell two enemies apart.
