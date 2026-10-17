// One replay as a row with a small map, its players and how it ended.

import Link from "next/link";

import { formatDuration } from "@/lib/format.ts";
import type { GameData, MatchRow } from "@/lib/types.ts";

import { MapPreview } from "./MapPreview.tsx";

export function MatchCard({ game, row, botNames }: { game: GameData; row: MatchRow; botNames: Record<string, string> }) {
  const map = game.maps.find((entry) => entry.id === row.map);
  const colors = Object.fromEntries(game.seats.map((seat) => [seat.id, seat.color]));
  const winner = row.ranking[0]?.length === 1 ? row.ranking[0][0] : null;
  return (
    <Link href={`/games/${game.id}/matches/${row.id}`} className="card match-card">
      <div className="match-card-art">{map && <MapPreview game={game} map={map} />}</div>
      <div className="match-card-body">
        <strong>{map?.name ?? row.map}</strong>
        <span className="match-players">
          {Object.entries(row.lineup).map(([seat, bot]) => (
            <span key={seat} className={`match-player${winner === seat ? " is-winner" : ""}`}>
              <span className="seat-dot" style={{ background: colors[seat] }} />
              {botNames[bot] ?? bot}
            </span>
          ))}
        </span>
        <span className="small">
          {winner ? `${botNames[row.lineup[winner]] ?? winner} won` : "Draw"} after {formatDuration(row.ticks * game.tick_seconds)}
        </span>
      </div>
    </Link>
  );
}
