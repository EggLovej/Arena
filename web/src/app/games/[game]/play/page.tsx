import type { Metadata } from "next";

import { PlayGame } from "@/components/PlayGame.tsx";
import { gameView } from "@/games/index.ts";
import { getGame, getLeaderboard } from "@/lib/data.ts";

type Params = { params: Promise<{ game: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { game } = await params;
  return { title: `Play ${getGame(game).name}` };
}

export default async function Play({ params }: Params) {
  const { game: id } = await params;
  const game = getGame(id);
  if (!gameView(id).play) {
    return <p className="notice">{game.name} can't be played on the site yet, only watched.</p>;
  }
  const tables = Object.values(getLeaderboard(id)?.tables ?? {});
  const ratings = Object.fromEntries(
    tables.map((table) => [String(table.players), Object.fromEntries(table.entries.map((entry) => [entry.bot, entry.rating]))]),
  );
  const thinking: Record<string, number> = {};
  for (const entry of tables.flatMap((table) => table.entries)) thinking[entry.bot] = Math.max(thinking[entry.bot] ?? 0, entry.think_ms_mean);
  return <PlayGame game={game} ratings={ratings} thinking={thinking} />;
}
