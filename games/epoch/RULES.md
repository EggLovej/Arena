Epoch is a duel between two bases at the ends of one lane, after the Flash game
Age of War: its five ages, its prices and its rewards. Each side trains units
that march along the lane toward the enemy base and fight whatever they meet.
Destroy the enemy base to win. If both fall at the same moment the match is a
draw. If neither has fallen after twenty minutes, the base with more hit points
left wins, and if they have as many, the side with more experience.

## Gold, experience and the ages

- Each side starts in the Stone Age with 200 gold and a base of 500 hit points.
- Killing an enemy unit, with a unit, a turret or a special, earns 130% of its
  price in gold. It also earns 400% of its price in experience (XP), and the
  side that lost it earns 50%.
- A little gold also comes in every second: 2 in the Stone Age, 5 in the Middle
  Ages, 20 in the Renaissance, 150 in the Modern Age and 500 in the Future.
- An age takes only experience: 4000 XP for the Middle Ages, 14000 for the
  Renaissance, 45000 for the Modern Age and 200000 for the Future. Each age
  brings new units, turrets and a special, and the base gains hit points: up
  to 1300, 3500, 12000 and 30000. Units already trained or waiting keep their
  age.

## Units

Every age has a melee, a ranged and a heavy unit, and the Future a super
soldier besides. Prices are Age of War's, in gold:

- **Stone Age**: Clubman (melee, 15), Slinger (ranged, 25), Mammoth rider
  (heavy, 100).
- **Middle Ages**: Knight (melee, 50), Crossbowman (ranged, 75), Catapult
  (heavy, 500).
- **Renaissance**: Duelist (melee, 200), Musketeer (ranged, 400), Cannon
  (heavy, 1000).
- **Modern Age**: Commando (melee, 1500), Gunner (ranged, 2000), Tank (heavy,
  7000).
- **Future**: Blade droid (melee, 5000), Laser trooper (ranged, 6000), Mech
  (heavy, 20000) and Super soldier (150000).

Up to five units can wait at a base. They train one at a time, which takes one
to six seconds, and the trained unit leaves as soon as there is room behind the
last one out.

## Marching and fighting

- Units march in single file. A unit stops when an enemy unit or the enemy base
  is in its reach, or when the friend in front is in the way.
- Melee and heavy units have to come close; ranged units hit from 100 to 140
  away, so they shoot past the friends in front of them. A lane is 1000 long on
  the Meadow, 700 in the Narrow pass and 1400 on the Frontier, and each base
  takes up 120 of it.
- When it is ready, a unit strikes the enemy base if the base is in its reach,
  and otherwise the nearest enemy unit.

## Turrets

- Each age has three turrets, a cheap one, a quick one and a strong one with a
  long reach: from the Rock thrower (100 gold) to the Ion ray (100000). A
  turret strikes the nearest enemy unit within its reach of the base, about 180
  to 300.
- A base has four places for turrets. The first is open from the start;
  opening the others costs 1000, 3000 and 7500 gold.
- Selling a turret returns half its price and leaves its place open. It sells
  the cheapest.

## Specials

Each age has a special: Rock rain, Arrow storm, Healing, Air strike and Orbital
laser. It is free, but a side can call one only every 45 seconds. Healing
restores every unit of its side on the field; the others strike every enemy
unit on the field at once.

## Ticks

A tick is a tenth of a second. Each tick, both sides see the same state and
choose at most one action: train a melee, ranged, heavy or super unit, build
one of the age's turrets, open a turret place, sell a turret, advance the age or
call the special. Then gold comes in, trained units leave their bases,
everything in reach strikes at once, and units march. Both armies strike and
march from where things stood before anyone moved, so neither side gains from
acting first.

## In the Arena

Every match on every lane is played from both seats. A bot that raises an
error, or returns something that isn't an action, does nothing that tick and
plays on. Nothing in Epoch is hidden or left to chance: a bot sees both sides'
gold, XP, queues, turrets and every unit.
