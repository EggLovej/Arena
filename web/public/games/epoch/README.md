# Epoch artwork

`sprites.svg` holds every unit, turret and base as a `<symbol>`; `lane.svg` is
the backdrop (1000 × 300, ground line at y = 250, `xMidYMax slice`).
`web/src/games/epoch/EpochBoard.tsx` draws them; this file says how they are
made, for adding ages and units.

## Symbols

| Age | melee | ranged | heavy | turret | base |
| --- | --- | --- | --- | --- | --- |
| `stone` | `clubman` | `slinger` | `mammoth_rider` | `rock_thrower` | `base_stone` |
| `medieval` | `knight` | `crossbowman` | `catapult` | `arrow_tower` | `base_medieval` |
| `future` | `blade_droid` | `laser_trooper` | `mech` | `laser_turret` | `base_future` |

The ids are the ones in `games/epoch/epoch/rules.py`. A unit has three frames,
`<unit>-walk-1` (also its standing pose), `<unit>-walk-2` and `<unit>-attack`;
turrets and bases have one, named for themselves. The viewBox is the size in
board units: melee and ranged 48 × 48, heavy 80 × 64, turret 40 × 40, base
120 × 150. Every symbol faces right, with its feet (or wheels, or footing) on
the bottom edge, centred; the board mirrors them for the right-hand seat.
Team parts (clothes, banners, shields, plumes, visors, lights, muzzle flashes)
are `currentColor` and nothing else is, so the board sets CSS `color` to the
seat's colour. Each base has bare ledges for the three turrets, at (40, −40),
(40, −80) and (40, −120) from its bottom centre.

## Adding sprites later

New ages and units should match the first set:

- One `<symbol>` per frame, using the sizes above. Shared body parts go in
  `<defs>` and are reused with `<use>` (see the `stone-leg`, `knight-body` and
  `mech-leg` parts).
- Flat fills only: no gradients, filters, masks, clip paths or `<style>`,
  which some browsers drop through `<use>` of another file.
- Outline `#2b211c`, 1.5 wide (1 for small details), round joins and caps, set
  on each `<symbol>` (attributes on `<svg>` or `<defs>` do not carry through).
- Light from the upper left: one darker shade per material on the far side,
  and far-side limbs in the shade colour.
- Chunky proportions: a head is about a third of a person's height; limbs
  about 4.5 wide.
- Team colour only on team parts, everything else from the palette.

| Material | Fill | Shade |
| --- | --- | --- |
| Ink (outline, eyes) | `#2b211c` | |
| Skin | `#e8b48a` | `#c98e66` |
| Hair; cave and doorway dark | `#5a3a26` | `#3e281b` |
| Mammoth fur | `#8b5a3a` | `#6a4128` |
| Hide, leather | `#a8743f` | `#80552e` |
| Wood | `#b98a52` | `#8c6237` |
| Stone | `#a7a196` | `#7d776d` |
| Bone, ivory, fletching | `#f1e8d0` | |
| Steel | `#c3ccd3` | `#8a959e` |
| Gold trim | `#e4b440` | |
| Future panel | `#e7ecef` | `#b8c1c8` |
| Gunmetal | `#4c5662` | `#353c46` |
| Glint on visors and flashes | `#fff` | |
