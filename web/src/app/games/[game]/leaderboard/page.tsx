import type { Metadata } from "next";

import { HeadToHead, LeaderboardTable } from "@/components/LeaderboardTable.tsx";
import { getGame, getLeaderboard } from "@/lib/data.ts";

type Params = { params: Promise<{ game: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { game } = await params;
  return { title: `${getGame(game).name} leaderboard` };
}

export default async function LeaderboardPage({ params }: Params) {
  const { game: id } = await params;
  const game = getGame(id);
  const leaderboard = getLeaderboard(id);
  if (!leaderboard) return <p className="notice">No tournament has been played yet.</p>;
  const anchor = game.rating_anchor;
  const anchorName = anchor ? game.bots.find((bot) => bot.id === anchor[0])?.name ?? anchor[0] : null;
  const mapNames = Object.fromEntries(game.maps.map((map) => [map.id, map.name]));

  return (
    <>
      <p className="small">
        Ratings on the Elo scale{anchorName && anchor ? `, with ${anchorName} fixed at ${anchor[1]}` : ""}: 100 points more
        wins about 64% of games, 200 about 76%, 400 about 91%. The ± is a 95% interval. {leaderboard.matches} matches,
        updated {new Date(leaderboard.updated).toUTCString().slice(5, 22)} UTC.
      </p>
      {Object.values(leaderboard.tables).map((table) => (
        <section key={table.players}>
          <h2>{table.label}</h2>
          <p className="small">
            {table.matches} matches on {table.maps.map((map) => mapNames[map] ?? map).join(", ")}.
          </p>
          <LeaderboardTable table={table} gameId={id} />
          <h3>Head to head</h3>
          <p className="small">
            The share of points the bot on the left took from the bot above
            {table.players > 2 ? ", counting a match as a win for whoever finished ahead" : ""}.
          </p>
          <HeadToHead table={table} gameId={id} />
        </section>
      ))}
    </>
  );
}
