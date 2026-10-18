# Training and evaluation

## Baseline

Stable-Baselines3 PPO runs on CPU with four synchronous environments. The actor and critic each have two 128-unit Tanh layers. Defaults: rollout 1,024 steps/environment, batch 256, five optimization epochs, learning rate 0.0003, gamma 0.999, GAE lambda 0.995, clip 0.2, target KL 0.02 and entropy coefficient 0.005. These are starting settings, not tuned optima. Decisions and simulation both run at 30 Hz; no action repeat or observation/reward normalization wrapper is added.

Training samples Classic/Crossfire, both seats, and a 50/50 mixture of canonical spawns and legal jitter within 75 units with randomized aim. Up to 32 candidate placements are tried before falling back to canonical spawns. Episode seeds advance from persistent Gym RNG; starting with the same seed repeats the sequence.

Initially opponents are uniformly Random, Template, Direct, Bank Shot and Reactive Defender. Every 102,400 transitions a frozen checkpoint is added. When snapshots exist, half of episodes sample the latest five snapshots and half sample scripted bots. Opponents change only between episodes; loaded models are cached. The 8,192-transition smoke run checkpoints every 4,096 transitions to exercise the callback path.

## Commands and configuration

Install `learn` plus any display/dev extras required, or use the README's all-extras setup.

```bash
ricochet train --output runs/my_smoke --smoke
ricochet train --config configs/ppo_three_seeds.json --output runs/experiment
ricochet train --resume runs/experiment/seed_0/latest.zip --seed 10 --output runs/continued
ricochet evaluate runs/continued/best.zip --suite test --output results/test.json
```

A JSON config can override fields of `TrainConfig`; `--seed`, `--steps` and `--smoke` override relevant fields. `--seeds 0,1,2` or the supplied config launches separate seed directories serially. Step counts and checkpoint intervals must align to complete rollouts. Existing run directories are rejected to preserve lineage.

Resuming validates hyperparameters and artifact versions, keeps optimizer/model/counters, and starts fresh episodes with the requested seed. This is an explicit continuation boundary, not bit-identical mid-episode recovery. A resumed run gets a new directory and records its parent checkpoint path and hash.

## Artifacts

- `manifest.json`: resolved configuration, versions, code digest/revision, dependency versions, embedded map definitions, scripted opponent digests and parent lineage.
- `checkpoint_NNNNNNNNN.zip` plus `.json`: trusted SB3 policy/optimizer state and compatible rules/encoder/action metadata, with frozen opponent identities.
- `latest.zip` / `best.zip` plus matching metadata: copies of numbered checkpoints. Earlier checkpoints win validation-score ties.
- `metrics.jsonl`: PPO metrics, reward components, completed-episode outcomes, wall time/throughput, active opponent IDs and validation scores.
- `episodes.jsonl`: completed episode seed, map, player seat, selected opponent and outcome.
- `eval_NNNNNNNNN.json`: full fixed-suite validation results.

Numbered checkpoints and run configuration are not overwritten. Models and metadata are local trusted artifacts; arbitrary uploaded checkpoints are not supported.

## Evaluation boundaries

Validation uses Classic/Crossfire, all five scripted opponents, both seats and canonical plus ten fixed jitter scenarios (seeds 10000–10009): 220 matches. Checkpoint selection uses mean win + half-draw score, never shaped training return.

The final `test` suite uses held-out Channels and Close Range/Bank Shot/Reactive Defender across both seats and the same scenario schedule: 66 matches. Do not use this report to repeatedly choose checkpoints. Close Range is not a training opponent. `--quick` reduces evaluation to canonical starts against Direct and is only a pipeline check (four validation games or two test games).

Repeated seeds with unchanged deterministic bots/maps do not supply independent statistical evidence. Reports preserve per-match outcomes, faults, damage, blocks, shield activations, energy expenditure and action latency. Results against these opponents do not establish optimality. Attention, recurrent networks, hybrid actions, action repeat and broader adversarial evaluation remain future controlled experiments.
