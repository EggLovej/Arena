import type { Snapshot } from "@/lib/types.ts";

import type { GameView, SeatStat } from "../types.ts";
import { GraphiteBoard, viewBox, type BaseState, type FleetState, type GraphiteBoardData } from "./GraphiteBoard.tsx";
import { GraphiteControls } from "./GraphiteControls.tsx";
import { look } from "./look.ts";

function totals(snapshot: Snapshot): Record<string, { units: number; bases: number }> {
  const result: Record<string, { units: number; bases: number }> = {};
  const add = (seat: string) => (result[seat] ??= { units: 0, bases: 0 });
  for (const [owner, units] of Object.values((snapshot.bases ?? {}) as Record<string, BaseState>)) {
    if (!owner) continue;
    add(owner).units += units;
    add(owner).bases += 1;
  }
  for (const [owner, units] of Object.values((snapshot.fleets ?? {}) as Record<string, FleetState>)) {
    add(owner).units += units;
  }
  return result;
}

export const graphite: GameView = {
  Board: GraphiteBoard,
  genre: "Real-time strategy",
  cardMap: "crossroads",
  botIcons: ["aggressive_ai", "careful_ai", "strategic_ai", "neural_ai", "beast_ai", "atlas_clean_ai"],
  accent: "#d3a42f", // the paper theme's highlight gold
  liveMap: "crossroads", // Instinct and Apex trade the lead three times
  look,
  stats(_board, snapshot) {
    const result: Record<string, SeatStat[]> = {};
    for (const [seat, { units, bases }] of Object.entries(totals(snapshot))) {
      result[seat] = [
        { label: "units", value: units },
        { label: bases === 1 ? "base" : "bases", value: bases },
      ];
    }
    return result;
  },
  strength(_board, snapshot) {
    return Object.fromEntries(Object.entries(totals(snapshot)).map(([seat, { units }]) => [seat, units]));
  },
  strengthLabel: "Units, including fleets on the way",
  play: {
    Controls: GraphiteControls,
    viewBox: (board) => viewBox(board as GraphiteBoardData, false),
    help: [
      "Click one of your bases, then a base next to it: half its units go there.",
      "Shift-click the second base instead (or turn Stream on) to send it everything the first base has, every tick, until you shift-click the first base again.",
      "You send one fleet a tick, like the bots. To let go of a selected base, click it again or right-click. Escape does too, except in full screen, where it leaves full screen.",
      "With a keyboard, Tab moves between the bases and Enter acts as a click (Shift+Enter as a Shift-click).",
    ],
    map: "crossroads",
  },
};
