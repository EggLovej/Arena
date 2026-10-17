import type { Snapshot } from "@/lib/types.ts";

import type { GameView, SeatStat } from "../types.ts";
import { RicochetBoard, type StatusState } from "./RicochetBoard.tsx";
import { RicochetControls } from "./RicochetControls.tsx";
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
  play: {
    Controls: RicochetControls,
    help: [
      "W A S D move your tank (the arrow keys do too), and it aims at the pointer.",
      "Hold the left mouse button to fire. A shot bounces off the first wall it meets and stops at the second; after the bounce it can hit you too.",
      "Hold Space or the right mouse button to raise your shield: it blocks every shot, but you can't fire and you move slower while it is up.",
      "Firing, moving and the shield cost energy, which comes back over time. Three hits and a tank is out.",
      "On a touch screen you can aim and fire by touching the board, but moving needs a keyboard.",
    ],
    map: "crossfire",
    pauseKey: "p", // Space raises the shield, as in the game's own window, and browsers keep Esc for leaving full screen
  },
};
