Graphite is played on a map of bases joined by routes. Every player starts with
one or more bases; the rest are neutral. The last player with any bases or
fleets left wins. A match that is still undecided after three minutes of game
time is a draw between everyone still in it.

## Bases and fleets

- An owned base produces units all the time, at the rate on its badge (for
  example `+1.5/s`). Neutral bases don't produce.
- Each tick, a player may send one fleet: a whole number of units from one of
  its bases along a route to a neighbouring base. The fleet takes the route's
  travel time to arrive (the label on the route).
- A fleet arriving at its own side's base reinforces it. Arriving anywhere else,
  it fights.

## Ticks

A tick is 0.1 seconds of game time, and every tick runs in the same order:

1. Every player sees the same state and chooses at most one action.
2. All valid actions are applied at once and their fleets depart together.
3. Owned bases produce units.
4. Fleets move.
5. All fleets arriving at one base are resolved together.

## Fights

All units at a base, the defenders and every arriving fleet, are pooled by
owner. The largest force wins and keeps the difference between itself and the
second largest. If the two largest are equal, the base becomes neutral with no
units. Because arrivals are resolved together, the order in which players act
never matters.

## For bots

A bot sees the whole board: every base, route and fleet, its own seat and the
time. It never sees map names, so it has to read the graph rather than
recognise a map. Two-player maps seat Red and Blue; three-player maps add Green.
