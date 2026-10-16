import type { Metadata } from "next";
import Link from "next/link";

import { LeaderboardTable } from "@/components/LeaderboardTable.tsx";
import { MatchCard } from "@/components/MatchCard.tsx";
import { featuredMatches } from "@/lib/featured.ts";
import { botNames, getGame, getLeaderboard, getMatches } from "@/lib/data.ts";

type Params = { params: Promise<{ game: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { game } = await params;
  return { title: getGame(game).name };
}

export default async function GameOverview({ params }: Params) {
  const { game: id } = await params;
  const game = getGame(id);
  const leaderboard = getLeaderboard(id);
  const matches = getMatches(id);
  const featured = featuredMatches(matches, leaderboard);
  const names = botNames(game);
  const tables = leaderboard ? Object.values(leaderboard.tables) : [];

  return (
    <div className="split">
      <section>
        <div className="section-title">
          <h2>Featured matches</h2>
          <Link href={`/games/${id}/watch`}>All replays</Link>
        </div>
        {featured.length ? (
          <div className="featured">
            {featured.map((row) => (
              <MatchCard key={row.id} game={game} row={row} botNames={names} />
            ))}
          </div>
        ) : (
          <p className="notice">No tournament has been played yet, so there are no replays.</p>
        )}
      </section>
      <section>
        {tables.map((table) => (
          <div key={table.players}>
            <div className="section-title">
              <h2>Top bots, {table.label}</h2>
              <Link href={`/games/${id}/leaderboard`}>Full table</Link>
            </div>
            <LeaderboardTable table={table} gameId={id} limit={5} />
          </div>
        ))}
        {leaderboard && (
          <p className="small">
            {leaderboard.matches} matches, updated {new Date(leaderboard.updated).toUTCString().slice(5, 16)}.
          </p>
        )}
      </section>
    </div>
  );
}
