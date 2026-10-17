// Which replays a game's overview shows first: strong bots against each other,
// each match on a different map and between a different set of bots, with up
// to two from the largest maps.

import type { Leaderboard, MatchRow } from "./types.ts";

export function featuredMatches(rows: MatchRow[], leaderboard: Leaderboard | null, count = 6): MatchRow[] {
  const replays = rows.filter((row) => row.replay);
  if (!leaderboard) return replays.slice(0, count);
  const rank: Record<string, number> = {};
  for (const table of Object.values(leaderboard.tables)) {
    table.entries.forEach((entry, index) => {
      rank[entry.bot] = Math.min(rank[entry.bot] ?? Infinity, index);
    });
  }
  // A match is as interesting as its weakest player is strong; decided matches first.
  const strength = (row: MatchRow) => Math.max(...Object.values(row.lineup).map((bot) => rank[bot] ?? 99));
  const sorted = [...replays].sort(
    (a, b) => strength(a) - strength(b) || Number(a.ranking[0].length > 1) - Number(b.ranking[0].length > 1) || a.id.localeCompare(b.id),
  );
  const size = (row: MatchRow) => Object.keys(row.lineup).length;
  const largest = Math.max(0, ...sorted.map(size));
  const bots = (row: MatchRow) => Object.values(row.lineup).sort().join(" ");

  const chosen: MatchRow[] = [];
  const maps = new Set<string>();
  const lineups = new Set<string>();
  const take = (pool: MatchRow[], limit: number, fresh: boolean) => {
    let taken = 0;
    for (const row of pool) {
      if (chosen.length >= count || taken >= limit) return;
      if (maps.has(row.map) || (fresh && lineups.has(bots(row)))) continue;
      chosen.push(row);
      maps.add(row.map);
      lineups.add(bots(row));
      taken++;
    }
  };
  if (largest > 2) {
    take(sorted.filter((row) => size(row) === largest), 2, true);
    take(sorted.filter((row) => size(row) < largest), count, true);
  }
  take(sorted, count, true);
  take(sorted, count, false); // too few different lineups: allow repeats
  return chosen.sort((a, b) => size(a) - size(b) || strength(a) - strength(b) || a.id.localeCompare(b.id));
}
