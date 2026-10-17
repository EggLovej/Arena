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

export type PlayControlsProps = {
  board: unknown;
  snapshot: Snapshot; // the position as it stands
  theme: unknown;
  seat: string; // the person's seat
  person: unknown; // what the game says about the person's seat beyond the board (its Person.view())
  active: boolean; // false while paused, counting down or over: nothing to send then
  send: (command: unknown) => void; // a command for the game's Person (platform/arena/live.py)
  tools: HTMLElement | null; // a place in the row under the board for the game's own buttons (fill it with a portal)
};

/** What a game adds so that people can play it on the site (the play page is shared). */
export type PlayView = {
  /** Lies over the board while a person plays: takes their clicks and shows their selection and orders. */
  Controls: ComponentType<PlayControlsProps>;
  /** How to play, one short line each. */
  help: string[];
  /** The map the play page starts on (the first map if not given). */
  map?: string;
};

export type GameView = {
  Board: ComponentType<BoardProps>;
  /** Numbers shown beside each seat in the scoreboard, for example units and bases. */
  stats: (board: unknown, snapshot: Snapshot) => Record<string, SeatStat[]>;
  /** One number per seat for the chart under a replay: how well each side is doing. */
  strength: (board: unknown, snapshot: Snapshot) => Record<string, number>;
  strengthLabel: string;
  /** The map drawn on the game's card on the home page (the first map if not given). */
  cardMap?: string;
  /** For people to play the game against the bots; the game has no play page without it. */
  play?: PlayView;
};
