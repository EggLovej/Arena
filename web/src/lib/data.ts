// Reading the published data at build time. Every page is static, so these run
// only while `next build` generates the site.

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

import type { GameData, GameSummary, Leaderboard, MatchRow } from "./types.ts";

const DATA = path.join(process.cwd(), "public", "data");

function read<T>(...parts: string[]): T | null {
  const file = path.join(DATA, ...parts);
  if (!existsSync(file)) return null;
  return JSON.parse(readFileSync(file, "utf8")) as T;
}

export function listGames(): GameSummary[] {
  return read<GameSummary[]>("games.json") ?? [];
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
