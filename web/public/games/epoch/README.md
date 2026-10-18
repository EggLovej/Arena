# Epoch artwork

`sprites.svg` holds every unit, turret and base as a `<symbol>`; `lane.svg` is
the backdrop (1000 × 300, ground line at y = 250, `xMidYMax slice`).
`web/src/games/epoch/EpochBoard.tsx` draws them, including where each shooter's
shots leave it; this file says how they are made, for adding ages and units.

## Symbols

| Age | melee | ranged | heavy | turrets, cheapest first | base |
| --- | --- | --- | --- | --- | --- |
| `stone` | `clubman` | `slinger` | `mammoth_rider` | `rock_thrower`, `sling_nest`, `boulder_catapult` | `base_stone` |
| `medieval` | `knight` | `crossbowman` | `catapult` | `arrow_tower`, `ballista`, `oil_cauldron` | `base_medieval` |
| `renaissance` | `duelist` | `musketeer` | `cannon` | `small_cannon`, `large_cannon`, `mortar` | `base_renaissance` |
| `modern` | `commando` | `gunner` | `tank` | `machine_gun`, `rocket_launcher`, `twin_gun` | `base_modern` |
| `future` | `blade_droid` | `laser_trooper` | `mech`, and the `super_soldier` | `laser_turret`, `plasma_cannon`, `ion_ray` | `base_future` |

The ids are the ones in `games/epoch/epoch/rules.py`. A unit has three frames,
`<unit>-walk-1` (also its standing pose), `<unit>-walk-2` and `<unit>-attack`;
turrets and bases have one, named for themselves. The viewBox is the size in
board units: melee and ranged 48 × 48, heavy 80 × 64, the super soldier
64 × 64, turret 40 × 40 (drawn about 31 tall, to clear the ledge above), base
120 × 150. Every symbol faces right, with its feet (or wheels, or footing) on
the bottom edge, centred; the board mirrors them for the right-hand seat.
Team parts (clothes, banners, shields, plumes, visors, domes, doors, lights,
laser flashes) are `currentColor` and nothing else is, so the board sets CSS
`color` to the seat's colour. Each base has four bare ledges for turrets, at
(40, −36), (40, −72), (40, −108) and (40, −144) from its bottom centre.

## Adding sprites later

New ages and units should match this set:

- One `<symbol>` per frame, using the sizes above. Shared body parts go in
  `<defs>` and are reused with `<use>` (see the `stone-leg`, `knight-body` and
  `mech-leg` parts).
- Flat fills only: no gradients, filters, masks, clip paths or `<style>`,
  which some browsers drop through `<use>` of another file.
- Outline `#2b211c`, 1.5 wide (1 for small details), round joins and caps, set
  on each `<symbol>` (attributes on `<svg>` or `<defs>` do not carry through).
- Light from the upper left: one darker shade per material on the far side,
  and far-side limbs in the shade colour. Shading on team parts is black at low
  opacity, so any mid-tone seat colour works.
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
| Bronze (cannons) | `#c48f3e` | `#956a2b` |
| Plaster | `#eadcbc` | `#cbb78f` |
| Terracotta roof | `#c4673f` | `#9a4c2d` |
| Olive (modern kit) | `#7d8350` | `#5d623a` |
| Sandbag | `#cdb487` | |
| Concrete | `#bdb9b0` | `#918c83` |
| Future panel | `#e7ecef` | `#b8c1c8` |
| Gunmetal | `#4c5662` | `#353c46` |
| Fire, muzzle flash | `#f6a13a` | core `#ffe28a` |
| Gun smoke (no outline) | `#e9e3d8` | |
| Glint on visors and flashes | `#fff` | |
