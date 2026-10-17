import type { Snapshot } from "@/lib/types.ts";

import type { GameView, SeatStat } from "../types.ts";
import { RicochetBoard, type StatusState } from "./RicochetBoard.tsx";
import { look } from "./look.ts";

function statuses(snapshot: Snapshot): Record<string, StatusState> {
  return (snapshot.status ?? {}) as Record<string, StatusState>;
}

export const ricochet: GameView = {
  Board: RicochetBoard,
  genre: "Real-time tank duel",
  cardMap: "crossfire",
  accent: "#a657ff", // Neon Circuit's violet
  liveMap: "crossfire",
  liveSpeed: 1, // a duel is quick: in real time it is still over in seconds
  look,
  botIcons: ["bank_shot_ai", "close_range_ai", "direct_ai", "random_ai", "reactive_defender_ai", "sentinel_ai"],
  stats(_board, snapshot) {
    const result: Record<string, SeatStat[]> = {};
    for (const [seat, [health]] of Object.entries(statuses(snapshot))) {
      // A destroyed tank has no numbers, so the scoreboard says it is knocked out. Energy
      // changes every moment, so the board shows it under each tank instead.
      if (health > 0) result[seat] = [{ label: "health", value: health }];
    }
    return result;
  },
  strength(_board, snapshot) {
    return Object.fromEntries(Object.entries(statuses(snapshot)).map(([seat, [health]]) => [seat, health]));
  },
  strengthLabel: "Health",
};
