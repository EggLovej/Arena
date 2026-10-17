# Sentinel — scripted AI experiment journal

## Initial design

Inspected the current helpers, generated-map round robin and exact engine update order. Existing changes are preserved. No ML or hidden engine state is used.

The initial candidate combines a visibility graph around wall corners, movement-energy-aware pursuit, quadratic interception, mirrored bank-shot proposals validated against swept-circle collisions, and a short-horizon movement search against current projectiles (including self-owned ricochets). Shield use is disabled by default, explicitly anticipating future nerfs. Close-range attacks test whether stationary reaction defenses can be pressured before a projectile becomes observable.

Evaluation will use the repository's `play_fixtures` runner, both seats, all bundled bots, authored maps, unseen generated maps, and shield-rule variants. Baseline wins alone do not establish optimal play. Results and rejected changes are appended below as experiments finish.

## Experiment 1: authored maps, all six bundled opponents, both seats

36 games: 26 wins / 10 draws / 0 losses, zero faults, mean action time averaged 1.27 ms across matches. Won all six games against Bank Shot, Close Range, Random and Template; Direct 2W/4D; Reactive Defender 6D. Raw initial result: `results/sentinel_v1_dev.json`.

Kept predictive dodging, navigation and energy-aware stopping. Identified a close-range stopping threshold (38 pixels) outside the 36-pixel contact range; same-tick shots travel 14 pixels and hit at radius 22, so this difference lets a defender react. Testing contact-range pressure next. No shields used by Sentinel.

## Experiment 2: contact pressure and energy interaction

Moving the stopping threshold alone did not change the 26W/10D result. A trace showed repeated shots consuming regenerated energy at 59px, preventing the final approach. Suppressing shots against every stationary target improved the record to 32W/2D/2L, but introduced two losses against Close Range. Rejected that blanket stationary-target heuristic. Next version applies patient contact pressure only after observing an actual enemy shield; this is behavioral state, not opponent-name knowledge. Raw result: `results/sentinel_v2_dev.json`.

## Experiment 3–4: selective pressure and alignment

Selective patience after observed shields restored the two Close Range wins: 34W/2D/0L. The remaining Classic draws came from diagonal contact separation (36.0000002px), letting a shot survive just long enough for the defender to shield. Retained a cardinal-axis staging point near stationary shield users, with movement capped to arrive exactly rather than repeatedly overlap/push. This is a close-range tactic tied to current tick/collision semantics, not a general proof that shield defense is beatable under arbitrary rules.

Version 4 authored suite: **36W/0D/0L**, zero faults, mean per-match action-time average 1.76ms; worst observed single call 21.3ms. Raw fixtures and source hashes: `results/sentinel_v4/authored_default.json`, `authored_report.json`. Started generated seeds 100–103 with default, expensive and disabled shields without changing this version during the run.

## Generated-map validation, version 4

Seeds 100–103 (48 games per shield preset): default 45W/0D/3L; nerfed 44W/1D/3L; disabled 44W/1D/3L. All losses were to Close Range; no faults. Shield independence holds, but the authored-map sweep masked a close-combat weakness. These seeds are now development data, not an untouched final holdout.

Testing an observed-attack standoff rule: keep roughly 150–190px against shooters to preserve reaction time for newly spawned bullets. The existing search only sees current bullets, so rushing armed opponents into point-blank range can lose before any dodge is possible. Contact pressure remains for observed stationary shield defense.

## Experiment 5: broad standoff rejected

On development seeds 100–103, default rules: 38W/9D/1L. Close Range improved to 8/8 wins, but Bank Shot fell to 4W/3D/1L and Direct to 2W/6D. The broad standoff rule prevents pressure on mobile opponents and is not retained unchanged.

Ablation on Classic plus generated seeds 20–21, three attacking opponents, both seats (18 games per variant): full v5 11W/7D/0L; no banks 9W/9D/0L; no dodge 8W/6D/4L. Predictive dodging is especially important against Close Range (full 6W, no dodge 2W/4L). Bank-shot evidence is modest/mixed but supports retaining it for coverage. Raw results: `results/sentinel_ablation/`.

## Experiment 6: retain selective standoff

Restricted standoff to observed advancing, close-range firing (derived from positions, velocities and fresh projectiles, never controller identity). This restores pressure against lateral/ranged opponents while leaving room to react to pursuers.

Final candidate on authored maps: 36W/0D/0L. Development generated seeds 100–103: default 48W/0D/0L; expensive shields 47W/1D/0L. Frozen the candidate for new generated seeds 1000–1005, all six opponents, both seats, three shield presets (216 games). This fresh batch is validation, not a target for another tuning cycle.

## Final frozen-candidate validation

Six fresh generated maps (1000–1005), all six bundled opponents, both seats:

| Rules | Wins | Draws | Losses |
| --- | ---: | ---: | ---: |
| Default shields | 70 | 2 | 0 |
| Activation 60, sustained drain 90/sec | 71 | 1 | 0 |
| Shield activation above maximum energy | 71 | 1 | 0 |

Total 212W/4D/0L across 216 validation games. All four draws were against Bank Shot. No controller faults. Sentinel requested no shields. The frozen candidate also won 36/36 authored-map games. This is evidence against the current opponent pool on six unseen layouts, not a proof of optimality or invincibility against stronger bots.

Mean action time averaged across matches was 2.65–3.84ms in the final parallel runs. Individual calls reached 102ms under concurrent load; no hard real-time bound is claimed. The visibility-graph setup and bank-shot checks are candidates for future latency optimization. Broader adversarial opponents and uncertainty-aware future-shot planning remain opportunities.

Final artifacts, exact rule values and controller source hashes are archived under `docs/verification_data/sentinel/`. The ablation report is explicitly version 5, before selective standoff; it is not a final-version ablation. Development seeds 100–103 were not counted as fresh validation.

Verification: all 152 tests pass (two upstream PettingZoo deprecation warnings), scoped Ruff passes, and whitespace checks pass. A recorded generated:1000 victory over Close Range replayed to the identical final state hash `017bc22214fc758338ad0cab10aa87466f94134e02af8799246f50c2958473ec`. Installed package refreshed; existing user work and game rules preserved.
