// Which replays a game's overview shows first: the strongest bots against each
// other, each on a different map, and a match on the largest maps.

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
  const chosen: MatchRow[] = [];
  const maps = new Set<string>();
  const sizes = new Set<number>();
  for (const row of sorted) {
    const size = Object.keys(row.lineup).length;
    if (maps.has(row.map)) continue;
    if (sizes.has(size) && size > 2 && chosen.filter((match) => Object.keys(match.lineup).length === size).length >= 2) continue;
    chosen.push(row);
    maps.add(row.map);
    sizes.add(size);
    if (chosen.length >= count) break;
  }
  return chosen;
}
