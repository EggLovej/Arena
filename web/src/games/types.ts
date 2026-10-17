// What every game brings to the website: a board that draws one moment of a
// match, and a few numbers per seat for the scoreboard. Everything else (the
// replay player, match lists, leaderboards, bot cards, rules) is shared.

import type { ComponentType } from "react";

import type { Seat, Snapshot } from "@/lib/types.ts";

export type BoardProps = {
  board: unknown; // the replay's or map's "board": what never changes in a match
  snapshot: Snapshot; // what to draw for this tick
  tick: number; // may have a fraction, so moving things move smoothly between ticks
  tickSeconds: number;
  theme: unknown; // the game's theme JSON
  seats: Seat[];
  compact?: boolean; // a small preview: shapes only, no labels
};

export type SeatStat = { label: string; value: number };

export type GameView = {
  Board: ComponentType<BoardProps>;
  /** Numbers shown beside each seat in the scoreboard, for example units and bases. */
  stats: (board: unknown, snapshot: Snapshot) => Record<string, SeatStat[]>;
  /** One number per seat for the chart under a replay: how well each side is doing. */
  strength: (board: unknown, snapshot: Snapshot) => Record<string, number>;
  strengthLabel: string;
  /** A few words under the game's name in the game list, for example "Real-time strategy". */
  genre: string;
  /** The map drawn beside the game's name in the game list (the first map if not given). */
  cardMap?: string;
  /** The glow around the game's board on its overview (the site's accent if not given). */
  accent?: string;
};
