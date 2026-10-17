"use client";

// The replays of the last tournament, filtered by map or bot.

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useMemo, useState } from "react";

import { formatDuration } from "@/lib/format.ts";
import type { MatchRow, Seat } from "@/lib/types.ts";

type Props = {
  gameId: string;
  rows: MatchRow[];
  seats: Seat[];
  botNames: Record<string, string>;
  mapNames: Record<string, string>;
  tickSeconds: number;
};

export function MatchList({ gameId, rows, seats, botNames, mapNames, tickSeconds }: Props) {
  const params = useSearchParams();
  const [map, setMap] = useState(params.get("map") ?? "");
  const [bot, setBot] = useState(params.get("bot") ?? "");
  const [opponent, setOpponent] = useState(params.get("against") ?? "");
  const colors = Object.fromEntries(seats.map((seat) => [seat.id, seat.color]));

  const shown = useMemo(
    () =>
      rows.filter((row) => {
        const bots = Object.values(row.lineup);
        return (!map || row.map === map) && (!bot || bots.includes(bot)) && (!opponent || bots.includes(opponent));
      }),
    [rows, map, bot, opponent],
  );
  const maps = [...new Set(rows.map((row) => row.map))];
  const bots = [...new Set(rows.flatMap((row) => Object.values(row.lineup)))].sort((a, b) =>
    (botNames[a] ?? a).localeCompare(botNames[b] ?? b),
  );

  return (
    <div className="match-list">
      <div className="filters">
        <label>
          Map
          <select value={map} onChange={(event) => setMap(event.target.value)}>
            <option value="">Every map</option>
            {maps.map((id) => (
              <option key={id} value={id}>
                {mapNames[id] ?? id}
              </option>
            ))}
          </select>
        </label>
        <label>
          Bot
          <select value={bot} onChange={(event) => setBot(event.target.value)}>
            <option value="">Any bot</option>
            {bots.map((id) => (
              <option key={id} value={id}>
                {botNames[id] ?? id}
              </option>
            ))}
          </select>
        </label>
        <label>
          Against
          <select value={opponent} onChange={(event) => setOpponent(event.target.value)}>
            <option value="">Anyone</option>
            {bots
              .filter((id) => id !== bot)
              .map((id) => (
                <option key={id} value={id}>
                  {botNames[id] ?? id}
                </option>
              ))}
          </select>
        </label>
        <span className="filters-count">
          {shown.length} {shown.length === 1 ? "replay" : "replays"}
        </span>
      </div>
      <ul className="matches">
        {shown.map((row) => {
          const winner = row.ranking[0]?.length === 1 ? row.ranking[0][0] : null;
          return (
            <li key={row.id}>
              <Link href={`/games/${gameId}/matches/${row.id}`} className="match-row">
                <span className="match-map">{mapNames[row.map] ?? row.map}</span>
                <span className="match-players">
                  {Object.entries(row.lineup).map(([seat, id]) => (
                    <span key={seat} className={`match-player${winner === seat ? " is-winner" : ""}`}>
                      <span className="seat-dot" style={{ background: colors[seat] }} />
                      {botNames[id] ?? id}
                    </span>
                  ))}
                </span>
                <span className="match-result">
                  {winner ? `${botNames[row.lineup[winner]] ?? winner} won` : "Draw"} · {formatDuration(row.ticks * tickSeconds)}
                </span>
              </Link>
            </li>
          );
        })}
      </ul>
      {!shown.length && <p className="notice">No replays match these filters.</p>}
    </div>
  );
}
