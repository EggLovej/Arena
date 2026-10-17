import type { Snapshot } from "@/lib/types.ts";

import type { GameView, SeatStat } from "../types.ts";
import { GraphiteBoard } from "./GraphiteBoard.tsx";

type BaseState = [string, number];
type FleetState = [string, number, string, string, number, number];

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
  cardMap: "crossroads",
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
};
