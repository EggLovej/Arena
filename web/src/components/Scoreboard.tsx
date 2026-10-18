// The seats of a match with the game's numbers for each, and the clock: the same
// above a replay and above a match played on the site.

import type { CSSProperties } from "react";

import type { SeatStat } from "@/games/types.ts";
import { formatClock } from "@/lib/format.ts";
import type { Seat } from "@/lib/types.ts";

import { BotName } from "./BotName.tsx";

type Props = {
  gameId: string;
  lineup: Record<string, string>; // seat to bot, in the order shown
  seats: Seat[];
  botNames: Record<string, string>;
  stats: Record<string, SeatStat[]>; // the game's numbers (GameView.stats); a seat without any is knocked out
  ranking: string[][] | null; // once the match is over: every seat behind first place is greyed out
  you?: string | null; // the person's seat, if they play
  seconds: number; // game time
  note: string; // under the clock
};

export function Scoreboard({ gameId, lineup, seats, botNames, stats, ranking, you, seconds, note }: Props) {
  const seatInfo = Object.fromEntries(seats.map((seat) => [seat.id, seat]));
  const order = Object.keys(lineup);
  return (
    <div className="scoreboard" style={{ "--seats": order.length } as CSSProperties}>
      {order.map((seat) => {
        const out = ranking !== null && ranking.findIndex((group) => group.includes(seat)) > 0;
        return (
          <div key={seat} className={`scoreboard-seat${out ? " is-out" : ""}${seat === you ? " is-you" : ""}`} style={{ "--seat": seatInfo[seat]?.color } as CSSProperties}>
            <span className="scoreboard-name">{seat === you ? "You" : <BotName name={botNames[lineup[seat]] ?? lineup[seat]} game={gameId} id={lineup[seat]} />}</span>
            <span className="scoreboard-color">{seatInfo[seat]?.name ?? seat}</span>
            <span className="scoreboard-stats">
              {stats[seat]?.map((stat) => (
                <span key={stat.label}>
                  <strong>{stat.value}</strong> {stat.label}
                </span>
              )) ?? <span>Knocked out</span>}
            </span>
          </div>
        );
      })}
      <div className="scoreboard-clock">
        <span>{formatClock(seconds)}</span>
        <small>{note}</small>
      </div>
    </div>
  );
}
