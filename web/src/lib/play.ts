// Playing a game against bots in the browser: what the play worker (public/workers/play.js)
// sends, and the small sums the play page does with it. The worker runs the match in
// Python (platform/arena/live.py); the page draws it and sends the person's commands.

import type { Delta, Snapshot } from "./types.ts";

/** Where Python for the browser (Pyodide) comes from. Set NEXT_PUBLIC_PYODIDE_URL to serve it from elsewhere. */
export const PYODIDE_URL = process.env.NEXT_PUBLIC_PYODIDE_URL || "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";

/** Stands in for a bot id where a person plays a seat, as PERSON in platform/arena/live.py. */
export const PERSON = "person";

export type Outcome = { ranking: string[][]; reason: string; ticks: number };

export type MatchStart = {
  board: unknown;
  snapshot: Snapshot;
  tick: number;
  people: Record<string, unknown>;
};

export type MatchUpdate = {
  tick: number;
  delta: Delta;
  refused: Record<string, string>; // why a person's action was refused this tick, by seat
  thinking_ms: Record<string, number>; // each bot's last move
  errors: Record<string, string>; // the first error of each bot that has had one
  outcome: Outcome | null; // once the match is over
  people: Record<string, unknown>;
};

/** `id` is the page's id for the match a message is about; a load error has none. */
export type WorkerMessage =
  | { type: "progress"; text: string }
  | { type: "loaded"; seconds: number }
  | { type: "started"; id: number; start: MatchStart }
  | { type: "tick"; id: number; update: MatchUpdate; pace: number }
  | { type: "refused"; id: number; reason: string }
  | { type: "error"; id?: number; message: string };

/**
 * Where to draw the match between ticks, so fleets glide: the last tick played plus the
 * share of the next one that has passed, never more than one tick ahead.
 */
export function drawnTick(tick: number, playedAt: number, now: number, tickMs: number, moving: boolean): number {
  if (!moving || tickMs <= 0) return tick;
  return tick + Math.min(1, Math.max(0, (now - playedAt) / tickMs));
}

/** How a match went for one seat: "won", "lost" or "drew". */
export function result(outcome: Outcome, seat: string): "won" | "lost" | "drew" {
  const [first = []] = outcome.ranking;
  if (!first.includes(seat)) return "lost";
  return first.length === 1 ? "won" : "drew";
}

/** A seat for every place on the map: the person in theirs, and the chosen bot (or the first choice) elsewhere. */
export function lineupFor(seats: string[], personSeat: string, bots: Record<string, string>, fallback: string): Record<string, string> {
  return Object.fromEntries(seats.map((seat) => [seat, seat === personSeat ? PERSON : bots[seat] || fallback]));
}
