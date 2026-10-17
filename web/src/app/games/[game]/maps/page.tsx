import type { Metadata } from "next";
import Link from "next/link";

import { MapPreview } from "@/components/MapPreview.tsx";
import { getGame } from "@/lib/data.ts";
import type { GameData, MapEntry } from "@/lib/types.ts";

type Params = { params: Promise<{ game: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { game } = await params;
  return { title: `${getGame(game).name} maps` };
}

export default async function Maps({ params }: Params) {
  const { game: id } = await params;
  const game = getGame(id);
  const seatNames = Object.fromEntries(game.seats.map((seat) => [seat.id, seat.name]));
  // The main list first, then each group under its own heading, such as Graphite's large maps.
  const groups = [...new Set(game.maps.map((map) => map.group ?? ""))].sort((a, b) => (a ? 1 : 0) - (b ? 1 : 0));
  return groups.map((group) => (
    <section key={group} className="map-group">
      {group && <h2>{group}</h2>}
      <MapCards game={game} maps={game.maps.filter((map) => (map.group ?? "") === group)} seatNames={seatNames} />
    </section>
  ));
}

function MapCards({ game, maps, seatNames }: { game: GameData; maps: MapEntry[]; seatNames: Record<string, string> }) {
  const id = game.id;
  return (
    <div className="map-cards">
      {maps.map((map) => (
        <Link key={map.id} href={`/games/${id}/watch?map=${map.id}`} className="card map-card game-card">
          <div className="map-card-art">
            <MapPreview game={game} map={map} />
          </div>
          <div className="map-card-body">
            <h2>{map.name}</h2>
            <p className="small">
              {map.seats.length} players ({map.seats.map((seat) => seatNames[seat] ?? seat).join(", ")}) · {map.description}
            </p>
          </div>
        </Link>
      ))}
    </div>
  );
}
