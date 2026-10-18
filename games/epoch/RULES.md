Epoch is a duel between two bases at the ends of one lane. Each side trains
units that march along the lane toward the enemy base and fight whatever they
meet. Destroy the enemy base to win. If both fall at the same moment the match
is a draw, and if neither has fallen after eight minutes, the base with more
hit points left wins.

## Gold, experience and the ages

- Each side starts in the Stone Age with 200 gold and a base of 500 hit points.
- Gold comes in every second: 8 in the Stone Age, 14 in the Middle Ages and 24
  in the Future.
- Killing an enemy unit earns 60% of its price in gold. If one of your units
  helped kill it, it also earns its full price in experience (XP); a kill by
  turrets alone earns no XP.
- Advancing to the Middle Ages takes 1000 XP and 500 gold; to the Future, 3000
  XP and 1500 gold. Each age brings new units and a new turret, and the base
  gains hit points: up to 900 in the Middle Ages and 1500 in the Future.
  Units already trained or waiting keep their age.

## Units

Every age has three units, and each one counters another: melee units deal
150% damage to ranged ones, ranged to heavy, and heavy to melee. A heavy unit
also hits every enemy unit in its reach at once.

- **Stone Age**: Clubman (melee, 15 gold), Slinger (ranged, 28 gold), Mammoth
  rider (heavy, 120 gold).
- **Middle Ages**: Knight (melee, 40 gold), Crossbowman (ranged, 75 gold),
  Catapult (heavy, 300 gold).
- **Future**: Blade droid (melee, 100 gold), Laser trooper (ranged, 190 gold),
  Mech (heavy, 750 gold).

Up to five units can wait at a base. They train one at a time, which takes one
to four seconds, and the trained unit leaves as soon as there is room in front
of the base.

## Marching and fighting

- Units march in three files side by side, each in single file, so three can
  fight at the front of an army.
- A unit stops when an enemy unit or the enemy base is in its reach, or when a
  friend in its file is in the way. Melee and heavy units have to come close;
  ranged units hit from about 100 away. A lane is 1000 long on the Meadow, 700
  in the Narrow pass and 1400 on the Frontier, and each base takes up 120 of it.
- When it is ready, a unit strikes the enemy base if the base is in its reach,
  and otherwise the nearest enemy unit.

## Turrets

- A base has three places for turrets. The first is open from the start;
  opening the second costs 150 gold and the third 400, on top of the turret.
- A turret is the current age's: the Rock thrower (120 gold), the Arrow tower
  (300 gold) or the Laser turret (750 gold). It strikes the nearest enemy unit
  within its reach of the base, about 200 to 260.
- Selling a turret returns half its price and leaves its place open. It sells
  the turret of the oldest age.

## Ticks

A tick is a tenth of a second. Each tick, both sides see the same state and
choose at most one action: train a melee, ranged or heavy unit, advance the age,
build a turret or sell one. Then gold comes in, trained units leave their
bases, everything in reach strikes at once, and units march. Both armies strike
and march from where things stood before anyone moved, so neither side gains
from acting first.

## In the Arena

Every match on every lane is played from both seats. A bot that raises an
error, or returns something that isn't an action, does nothing that tick and
plays on. Nothing in Epoch is hidden or left to chance: a bot sees both sides'
gold, XP, queues, turrets and every unit.
