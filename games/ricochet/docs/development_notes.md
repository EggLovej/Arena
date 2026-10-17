# Ricochet development notes

This journal records implementation checkpoints, attempted approaches, and why they were kept or rejected.

## 2026-10-10 — Project start

- Kept the Graph War principle that the rules engine must not import the renderer. Ricochet will expose immutable observations and accept simultaneous actions so visual, headless, tournament, and training modes all execute the same code.
- Kept bot discovery through an `ais/` directory, but changed the interface from a module-level function to `create_controller()`. Separate controller instances make mirror matches and stateful bots safe.
- Chose a standalone project with reusable internal contracts instead of extracting a shared multi-game framework. A framework can be extracted once a second game demonstrates which abstractions are truly common.
- The implementation is being assembled in milestones rather than written as a single final dump. Next checkpoint: deterministic game types, map validation, and engine physics.

## 2026-10-10 — Deterministic engine milestone

- Added frozen public values for actions, observations, tanks, projectiles, maps, match context, events, and step results. Mutable state remains private to the engine.
- Kept a fixed 30 Hz simulation and simultaneous action collection. Movement proposals are calculated before either tank is committed, and overlap correction is symmetric so dictionary iteration order cannot decide collisions.
- Used swept segment tests for projectile-to-wall and projectile-to-tank collisions. A simple end-position overlap test was rejected because a projectile travels 14 pixels per tick and could skip thin collision targets.
- Implemented walls as data-driven axis-aligned rectangles. This is intentionally less general than arbitrary polygons, but it makes map authoring, collision normals, reflected bank-shot calculations, and observation encoding predictable.
- Implemented three symmetric arenas. Their variety comes from route geometry rather than different rule sets, so tournament comparisons remain meaningful.
- Next checkpoint: controller loading, headless match faults/timing, replay verification, and tournament outputs.

## 2026-10-10 — Headless AI and evaluation milestone

- Added factory-based AI discovery. Loading the same file twice executes two factories, avoiding shared module-level match state.
- Controller exceptions and malformed actions become no-ops and are counted. Three faults forfeit the match. Hard process timeouts were not added because v1 explicitly trusts local bots; action latency is still measured for later competition infrastructure.
- Replays store validated actions and per-tick state hashes rather than full state snapshots. This keeps files small while making divergence explicit at the first incorrect tick.
- Tournaments swap seats for every bot, map, and seed combination and export both JSON and CSV. This prevents a single spawn side from silently deciding rankings.
- Added random, direct-fire, and bank-shot baselines plus a minimal template. The bank-shot bot uses the boundary reflection construction; it remains a baseline rather than a claim of optimal play.
- Next checkpoint: fixed-size training observations and the Gymnasium adapter.

## 2026-10-10 — Training interface milestone

- Added the planned 273-element observation: one clock feature, ten features per tank, twelve padded wall slots, and twenty-four padded projectile slots.
- Projectiles are sorted by distance to the learner with ID tie-breaking. Stable raw IDs alone were rejected because newly spawned low-threat projectiles could occupy prominent slots while immediate threats appeared late in the vector.
- Kept a six-value continuous action vector with thresholded fire and shield triggers. This works with common continuous-control policies while decoding to the exact typed action used by scripted bots and humans.
- The Gymnasium wrapper controls one seat against a factory-created opponent. The core engine still accepts both simultaneous actions, leaving self-play trainers free to bypass the single-agent wrapper.
- Reward shaping is limited to damage events plus terminal outcome. Positional or energy rewards were not added because they can teach proxies that conflict with winning.
- Next checkpoint: Pygame menu, human controller, match rendering, CLI wiring, and documentation.

## 2026-10-10 — Playable application milestone

- Added a keyboard-driven Pygame menu, a vector-drawn arena, HUD, human input, result/rematch flow, and visual replay playback. Pygame imports remain confined to `display/`.
- Human input implements the same controller protocol as bots. A separate event-only action path was rejected because it would let headless and visual rules drift.
- The CLI now routes menu, visual matches, headless matches, tournaments, and replays through the same engine and match session.
- Chose vector-drawn graphics instead of asset files for v1. This keeps the first version portable while still making aim, health, energy, shields, bounced shots, and obstacles readable.
- Added setup, architecture, controls, AI authoring, Gymnasium, and evaluation documentation.
- Next checkpoint: automated tests, dependency lock, tournament behavior checks, and visual smoke testing.

## 2026-10-11 — Validation and release milestone

- Locked dependencies with `uv` and validated the project under Python 3.12.14. The declared minimum remains Python 3.11 so contributors are not forced onto the current machine's newest interpreter.
- One initial uv editable-install file received a filesystem hidden flag, so Python skipped its `.pth` entry and the console script could not import the package. Rebuilding through uv cleared the flag. The wheel now also contains fallback copies of built-in maps and bots, while repository-local `maps/` and `ais/` take priority for contributors.
- `pytest` passes 23 tests covering determinism, simultaneous combat, energy, shields, bounces, self-damage, collision tunnelling, wall sliding, timeouts, map validation, architecture boundaries, controller instances and forfeits, replay parity, tournaments, vector encoding, seeded Gym resets, and the Gymnasium checker.
- `ruff check .` passes without exclusions for project code.
- The full acceptance match `direct_ai` versus `bank_shot_ai` on Classic ended after 424 ticks with Bank Shot winning 3 health to 0. Its replay reproduced the exact final state hash.
- The acceptance tournament ran 108 matches over all three maps, both seats, and seeds 0, 1, and 2. Scores were Bank Shot 45, Template 28.5, Direct 21, and Random 13.5. No bot faults occurred.
- These standings demonstrate that the evaluation path differentiates the bundled baselines; they do not establish that Bank Shot is optimal or robust against unfamiliar controllers.
- Pygame rendering passed a dummy SDL smoke test for a normal frame and result overlay. Interactive controls still require a real window for final human feel and tuning.

## 2026-10-11 — Result-screen flicker fix

- The result renderer previously called the normal frame renderer, which flipped the display, and then flipped again after drawing the overlay. The player therefore saw alternating plain-game and result frames.
- Split frame composition from presentation. Normal gameplay and the result screen now each compose all of their layers and call `pygame.display.flip()` exactly once.


## 2026-10-11 — Rules v2 implementation started

- Preserved the Python engine/display/training split and the existing controller lifecycle.
- Versioned the corrected rules; shots now sweep from the tank center, closing the unchecked muzzle gap. Visual muzzle placement remains a rendering concern.
- Added an upfront shield activation payment and six-tick prepaid commitment. The old shield economics remain a comparison preset, using corrected physics.
- Added complete state capture/restore/clone and optional verification hashing. Next: collision regression tests, environment semantics and replay provenance.

## 2026-10-11 — Runtime and training contracts

- Replays now embed full initial state/map/rules and store seek checkpoints every 300 ticks; v1 recordings are explicitly rejected because corrected physics changes outcomes.
- Added typed bot-module registration and startup-failure handling, including invalid matches when both controllers fail.
- Added defensive and pursuit evaluation bots, a Parallel API adapter, seeded episode variation, explicit external truncation and corrected health-loss shaping.
- Encoder v2 appends shield commitment, dimensions and relative opponent features. Capacity checks prevent silent information loss.
- The local editable-install issue recurred after clearing the macOS flag, so a fresh uv reinstall is being tested instead of depending on PYTHONPATH as the documented workaround.

## 2026-10-11 — PPO, visuals and first verification

- The 8,192-transition PPO smoke workflow completed, saved two checkpoints, reloaded and evaluated them. The quick validation score was 0.5 (draws); this demonstrates plumbing, not policy strength. Added a second smoke run after telemetry changes, preserving the earlier run rather than overwriting it.
- A checkpoint controller completed a normal recorded match, and the headless replay reproduced its final hash. Added tests for snapshot opponent selection, metadata incompatibility and optimizer/counter continuation across a short resume.
- Implemented the robot-sport presentation: separate HUD viewport, mechanical tank layers, beveled walls, bounded trails/sparks/ripples, independent cosmetic RNG, audio and mouse-selectable preview cards. No engine mutation is allowed from rendering.
- The controlled shield comparison ran 120 matches without faults. Draws changed from 30/60 to 28/60; the reactive defender still had 22 draws and 2 wins under each preset. The chosen cost improves this particular overall draw count only modestly and does not solve defensive dominance. Kept the specified new default and recorded the limitation instead of silently tuning against the evaluation set.
- Fresh core-only wheel and editable installations outside the checkout succeeded and imported none of Pygame/NumPy/Gymnasium/Torch. The local `.venv` repeatedly acquires hidden macOS flags on `.pth` files; clearing the flag was temporary. Adopted/documented a wheel-based local install to avoid reliance on `.pth` processing, while retaining standard editable packaging for unaffected environments.

## 2026-10-11 — Acceptance report

- All 50 tests pass, including a short optimizer-resume test and explicit snapshot-opponent sampling. Ruff and whitespace checks pass.
- Final smoke evaluation on held-out Channels: 66 valid games, 0 wins, 44 draws, 22 losses. This confirms test-suite execution but also that 8,192 training transitions are not a strong-policy result.
- Archived raw balance, benchmark and smoke-evaluation evidence under docs/verification_data and summarized limitations in docs/verification.md.
- Observed about 11.9k Gym steps/s without neural inference/optimization. Matched no-op traces measured 18.8k engine ticks/s with full hashes and 49.9k without; retained optional hashing rather than changing implementation language.
- Added fading short trails, segmented bounce interpolation, impact ripples and cooldown text after inspecting the rendered frames. Cosmetic changes remain independent of simulation state.
- Deferred hybrid policies, attention, recurrence, long training and further balance tuning as planned. No browser or public-bot infrastructure was introduced.


## 2026-10-11 — Selectable themes started

- Chose immutable, Pygame-independent theme definitions and separate backend drawing strategies. The renderer retains timing, interpolation, effects and presentation; styles own arena and tank appearance.
- Keeping the existing Robot Arena as a compatibility choice and adding Neon Circuit as the default. Theme selection belongs to visual sessions, never engine state, training observations or replay hashes.
- Building the neon geometry procedurally rather than using the generated concept as a flat background: walls must match every map and tanks must remain independently animated.

## 2026-10-11 — Theme integration and visual verification

- Added catalog-driven mouse/keyboard selection, CLI theme arguments and session retention through rematches/menu returns. Replay presentation uses the selected theme independently of recording state.
- Extracted existing floor/tank drawing into the industrial backend style; Neon Circuit supplies segmented circular robots, violet outlined walls, grid court, additive glow and angular HUD panels. Theme data imports only the standard library.
- Inspected actual menu and gameplay captures. Replaced visibly stepped halo rings with a smooth cached radial falloff and increased inactive menu affordance contrast. Kept procedural rendering for arbitrary maps and collision clarity.
- All 54 tests pass, including both themes across every map, resize/result presentation, immutable engine state, mouse/keyboard selection, CLI routing and dependency-free theme discovery. Two existing upstream PettingZoo deprecation warnings remain.


## 2026-10-11 — Neon proposal fidelity correction

- Compared the generated Neon Circuit proposal against the implemented screenshot. Identified old HUD hierarchy, dim hard outlines, thin aliased tank rings and missing corner decorations as the main discrepancies.
- Moving complete HUD/footer composition into style strategies, with tracked monospace typography, round health lights, full-width outlined energy bars and a centered game title/timer for Neon Circuit.
- Replacing offset outline rings with cached multiscale bloom and supersampling small layered tank sprites. Keeping actual map/collision dimensions rather than distorting the game to match the illustrative reference.

- Visually checked the real renderer with a reproducible staged scene against the proposal. Added `scripts/capture_theme_preview.py`, which explicitly labels arranged state as a visual fixture and captures both styles.
- The visual fixture exposed a shield fill overwriting the opaque tank sprite; moved the fill behind the tank and added a regression assertion for turret visibility.
- Retained actual map geometry, the existing viewport transform, one display flip per frame and simulation-independent styling. The generated reference's different arena aspect/obstacle proportions remain intentionally governed by the game map.
- Full suite passed before the final shield correction (54 tests); all 10 display tests passed after it. Ruff and whitespace checks pass. A 240-frame static dummy-display check averaged 1.19 ms/frame, not a live-window FPS guarantee. Rebuilt the local wheel install so the launcher uses the new renderer.
- Final verification after the shield correction: all 55 tests passed (two existing PettingZoo deprecation warnings).


## 2026-10-11 — Robot League implementation started

- Using the third proposal as the visual target: light worn concrete, metal perimeter rails, hazard stripes, charcoal steel blocks, cyan/coral tracked robots and soft shadows.
- Extracted the Neon Circuit HUD/footer into a shared layout mixin. Robot League changes materials and palette while inheriting the same HUD geometry; the original Robot Arena remains available as a legacy style.
- Generated two material textures with built-in imagegen and saved them in package data with their prompts/provenance. Keeping obstacles and robots independently rendered so every map and arbitrary turret orientation works.


## 2026-10-11 — Robot League visual refinement

- Inspected the actual renderer against the reference. Replaced the first procedural chassis with a detailed transparent neutral chassis asset, tinted armor at runtime and kept independently aimed turrets. Strengthened obstacle shadows and bevels, softened the concrete texture with a neutral wash, and retained subtle tread scuffs and perimeter lighting.
- Added a pixel comparison proving that equal palettes produce identical HUD output for Neon Circuit and Robot League. Extended all-map rendering/state-immutability checks to the new theme and updated selector/CLI checks.
- Generated alpha has near-transparent (1/255) background pixels, so the asset check accepts <=2/255 alpha at the outer corner and verifies an opaque chassis center. This avoids treating harmless antialias transparency as an opaque background.
- Acceptance: 57 tests passed, Ruff and whitespace checks clean. Inspected Robot League menu, Classic staged comparison and Crossfire gameplay render. HUD pixel-equivalence test passes; engine state remains unchanged under every theme/map render.
- Rebuilt the installed wheel and rendered Robot League from `/private/tmp`, verifying bundled textures/chassis load without the source checkout as working directory. Static 240-frame dummy-display measurement averaged 1.36 ms/frame (not a live-window FPS guarantee).


## 2026-10-11 — Robot League lighting and art-direction correction

- User identified sparse yellow paint, dim lamps and a weaker overall visual impression than the proposal. Expanded safety paint into broad corner/side bands and added yellow perimeter couplers and diagonal corner armor accents.
- Replaced barely visible alpha light ellipses with cached additive warm light pools, larger white-hot lamp cores and local bloom, on all four sides of the arena.
- Reduced distracting concrete microcontrast while retaining its material texture; added soft edge shading to establish a recessed court and contrast with the lights. HUD and gameplay geometry remain unchanged.
- Inspected the updated staged renderer preview: warm light spill is now visible on the concrete, safety markings frame all four sides, and softer floor detail improves object contrast. Refreshed menu/game/result captures and the installed wheel. All 57 tests pass; Ruff and whitespace checks are clean.
