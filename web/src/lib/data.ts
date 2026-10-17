// Reading the published data at build time. Every page is static, so these run
// only while `next build` generates the site.

import { readFileSync, statSync } from "node:fs";
import path from "node:path";

import { GAME_VIEWS } from "@/games/index.ts";

import type { GameData, GameSummary, Leaderboard, MatchRow } from "./types.ts";

const DATA = path.join(process.cwd(), "public", "data");

// Each file is parsed once, not once per page; a file that changes (during
// `next dev`) is read again. Pages must not change what they get.
const parsed = new Map<string, { modified: number; value: unknown }>();

function read<T>(...parts: string[]): T | null {
  const file = path.join(DATA, ...parts);
  const modified = statSync(file, { throwIfNoEntry: false })?.mtimeMs;
  if (modified === undefined) return null;
  const cached = parsed.get(file);
  if (cached?.modified === modified) return cached.value as T;
  const value = JSON.parse(readFileSync(file, "utf8")) as T;
  parsed.set(file, { modified, value });
  return value;
}

/** Every published game, in the order src/games/index.ts lists them. */
export function listGames(): GameSummary[] {
  const order = Object.keys(GAME_VIEWS);
  const place = (id: string) => (order.includes(id) ? order.indexOf(id) : order.length);
  return [...(read<GameSummary[]>("games.json") ?? [])].sort((a, b) => place(a.id) - place(b.id));
}

export function getGame(id: string): GameData {
  const game = read<GameData>(id, "game.json");
  if (!game) throw new Error(`No data for game ${id}: run "python -m arena publish ${id}"`);
  return game;
}

/** The last tournament's tables, or null before the first one has run. */
export function getLeaderboard(id: string): Leaderboard | null {
  return read<Leaderboard>(id, "leaderboard.json");
}

export function getMatches(id: string): MatchRow[] {
  return read<MatchRow[]>(id, "matches.json") ?? [];
}

export function botNames(game: GameData): Record<string, string> {
  return Object.fromEntries(game.bots.map((bot) => [bot.id, bot.name]));
}
