Ricochet is a duel between two tanks in a walled arena. Each tank starts with
three health and 100 energy. Destroy the other tank to win. If both are
destroyed at the same moment, or neither is after 90 seconds, the match is a
draw.

## Moving, aiming and firing

- A tank moves in any direction at up to 180 pixels a second and aims
  separately from the way it moves. It slides along walls, and two tanks push
  each other apart.
- Firing costs 32 energy, and a tank can fire at most twice a second. A shot
  flies straight at 420 pixels a second from the centre of the tank.
- A shot bounces off the first wall it meets and disappears at the second,
  or after five seconds. Until it bounces it can't hurt the tank that fired
  it; after the bounce it can.
- A shot that reaches a tank takes one health, unless the tank's shield is up.

## Energy and the shield

- Energy comes back at 12 a second, up to 100. Moving at full speed costs 16
  a second, so a tank that never stops moving soon has nothing left to fire
  with. Without enough energy, a tank moves more slowly.
- Raising the shield costs 23.2 energy at once and keeps it up for at least
  0.2 seconds. Holding it longer costs 36 energy a second.
- A raised shield blocks shots from every side, but the tank can't fire and
  moves at 60% speed.

## Ticks

A tick is a thirtieth of a second of game time. Both tanks see the same state
and choose an action: how to move, where to aim, whether to fire and whether to
shield. Then, in this order, energy comes back and shields go up or down, the
tanks move, they fire, and every shot moves and hits whatever it meets. Hits on
both tanks in the same tick count together.

## In the Arena

A bot that raises an error, or returns something that isn't an action, does
nothing that tick and plays on. Every match on every map is played from both
seats, so neither colour gains from a better starting spot. Besides the three
drawn maps, the Arena plays a few arenas made by Ricochet's map generator,
which are nearly but not exactly symmetric.
