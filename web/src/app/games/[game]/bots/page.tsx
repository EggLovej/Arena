import type { Metadata } from "next";
import Link from "next/link";

import { BotAvatar } from "@/components/BotAvatar.tsx";
import { BotName } from "@/components/BotName.tsx";
import { ScoreBar } from "@/components/LeaderboardTable.tsx";
import { getGame, getLeaderboard } from "@/lib/data.ts";

type Params = { params: Promise<{ game: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { game } = await params;
  return { title: `${getGame(game).name} bots` };
}

export default async function Bots({ params }: Params) {
  const { game: id } = await params;
  const game = getGame(id);
  const leaderboard = getLeaderboard(id);
  const tables = leaderboard ? Object.values(leaderboard.tables) : [];
  const mapNames = Object.fromEntries(game.maps.map((map) => [map.id, map.name]));
  const bestRating = (bot: string) => Math.max(-Infinity, ...tables.map((table) => table.entries.find((entry) => entry.bot === bot)?.rating ?? -Infinity));
  const bots = [...game.bots].sort((a, b) => bestRating(b.id) - bestRating(a.id));

  return (
    <div className="bot-cards">
      {bots.map((bot) => {
        const standings = tables.map((table) => ({ table, entry: table.entries.find((entry) => entry.bot === bot.id) }));
        const byMap = standings.flatMap(({ entry }) => Object.entries(entry?.by_map ?? {}));
        return (
          <article key={bot.id} id={bot.id} className="card bot-card">
            <div className="bot-card-head">
              <BotAvatar id={bot.id} kind={bot.kind} size={48} />
              <div>
                <h2>
                  <BotName name={bot.name} />
                </h2>
                <span className="tag">{bot.kind}</span> <code>{bot.id}</code>
              </div>
            </div>
            <p>{bot.description}</p>
            <dl className="bot-ratings">
              {standings.map(({ table, entry }) => (
                <div key={table.players}>
                  <dt>{table.label}</dt>
                  <dd>{entry ? entry.rating : "–"}</dd>
                </div>
              ))}
            </dl>
            {byMap.length > 0 && (
              <ul className="map-scores">
                {byMap.map(([map, { score, games }]) => (
                  <li key={map}>
                    <span>
                      {mapNames[map] ?? map} <span className="small">({games})</span>
                    </span>
                    <ScoreBar value={score} />
                  </li>
                ))}
              </ul>
            )}
            {bot.seat_counts && <p className="small">Plays only on {bot.seat_counts.join(" and ")}-player maps.</p>}
            <Link href={`/games/${id}/watch?bot=${bot.id}`}>Watch its matches</Link>
          </article>
        );
      })}
    </div>
  );
}
