import type { Metadata } from "next";
import { Suspense } from "react";

import { MatchList } from "@/components/MatchList.tsx";
import { botNames, getGame, getMatches } from "@/lib/data.ts";

type Params = { params: Promise<{ game: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { game } = await params;
  return { title: `Watch ${getGame(game).name}` };
}

export default async function Watch({ params }: Params) {
  const { game: id } = await params;
  const game = getGame(id);
  const rows = getMatches(id).filter((row) => row.replay);
  const mapNames = Object.fromEntries(game.maps.map((map) => [map.id, map.name]));
  return (
    <>
      <p className="small">
        One replay of every pairing on every map from the last tournament. Pick one to watch it play out.
      </p>
      {rows.length ? (
        <Suspense>
          <MatchList gameId={id} rows={rows} seats={game.seats} botNames={botNames(game)} mapNames={mapNames} tickSeconds={game.tick_seconds} />
        </Suspense>
      ) : (
        <p className="notice">No tournament has been played yet, so there are no replays.</p>
      )}
    </>
  );
}
