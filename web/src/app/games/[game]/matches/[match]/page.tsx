import type { Metadata } from "next";
import Link from "next/link";

import { ReplayPlayer } from "@/components/ReplayPlayer.tsx";
import { botNames, getGame, getMatches, listGames } from "@/lib/data.ts";

type Params = { params: Promise<{ game: string; match: string }> };

export const dynamicParams = false;

export function generateStaticParams() {
  return listGames().flatMap((game) =>
    getMatches(game.id)
      .filter((row) => row.replay)
      .map((row) => ({ game: game.id, match: row.id })),
  );
}

function describe(game: string, match: string) {
  const data = getGame(game);
  const row = getMatches(game).find((entry) => entry.id === match);
  const names = botNames(data);
  const players = row ? Object.values(row.lineup).map((bot) => names[bot] ?? bot) : [];
  const map = data.maps.find((entry) => entry.id === row?.map);
  return { data, row, names, players, map };
}

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { game, match } = await params;
  const { players, map } = describe(game, match);
  return { title: `${players.join(" vs ")} on ${map?.name ?? "?"}` };
}

export default async function MatchPage({ params }: Params) {
  const { game, match } = await params;
  const { data, row, names, players, map } = describe(game, match);
  return (
    <>
      <p className="small">
        <Link href={`/games/${game}/watch`}>Replays</Link> / {map?.name ?? row?.map}: {players.join(" vs ")}
      </p>
      <ReplayPlayer gameId={game} replayUrl={`/data/${game}/replays/${match}.json`} seats={data.seats} botNames={names} theme={data.theme} />
    </>
  );
}
