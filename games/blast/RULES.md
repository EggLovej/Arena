Two to four bombers start in the corners of a grid of stone walls and wooden
crates. They drop bombs that explode in a cross, break crates to find power-ups,
and try to catch each other in the flames. The last one standing wins.

## Walking and bombs

- A bomber steps one tile up, down, left or right. A step takes 0.3 seconds
  before the next can start. Walls, crates, bombs and other bombers are in the
  way, but a bomber can walk off a bomb it is standing on.
- A bomber drops a bomb on the tile it stands on. It starts with one bomb on
  the field at a time and gets it back when it goes off.
- A bomb goes off 2.5 seconds after it is dropped. Its flames reach two tiles in
  each of the four directions. They stop before a wall, and at the first crate
  or bomb they reach, which they burn.
- A bomb the flames reach goes off at once, so bombs set each other off in a
  chain.
- Flames burn for half a second. A bomber who is in them at any moment is out.
  Bombers caught at the same moment share their place.

## Crates and power-ups

- About seven in ten free floor tiles start with a crate, dealt again for every
  match. The tiles next to the starts are kept clear, and tiles that mirror each
  other across the map get the same, so starts that mirror each other have the
  same crates and power-ups around them.
- Four in ten crates hide a power-up, which appears when the crate burns:
  - **Bomb** (16 in 100): one more bomb on the field at a time, up to eight.
  - **Fire** (16 in 100): flames reach one tile further, up to eight.
  - **Speed** (8 in 100): quicker steps, from 0.3 seconds down to 0.25, 0.2 and
    0.15.
- A bomber picks a power-up up by stepping onto it. Flames burn a power-up
  lying in the open.

## The walls close in

After a minute and a half, the walls start closing in. Every 0.3 seconds one
more tile becomes wall, spiralling in from the edge to the middle, and anything
on it is crushed. The website marks the next few tiles before they close. A
match always ends before three minutes: whoever is left last wins, and the
rest are ranked by how long they lasted.

## Ticks

A tick is a twentieth of a second. Each tick every bomber still in the match
sees the whole field, except what the crates hide, and chooses to stay, step or
drop a bomb. Then, in this order, bombs are dropped and bombers step, each in
seat order (so of two bombers stepping onto the same tile, only the first gets
there); bombs whose time has come go off, with every bomb their flames reach;
anyone in the flames is out; and, once they have started, the walls close in by
a tile every sixth tick.

## In the Arena

A bot that raises an error, or returns something that isn't an action, does
nothing that tick and plays on. Every match is played from every seating, so no
bot gains from a better start.
