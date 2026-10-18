import type { Snapshot } from "@/lib/types.ts";

import type { GameView, SeatStat } from "../types.ts";
import { EpochBoard, HEIGHT, type EpochBoardData, type SideState } from "./EpochBoard.tsx";
import { EpochControls } from "./EpochControls.tsx";
import { look } from "./look.ts";

function sides(snapshot: Snapshot): Record<string, SideState> {
  return (snapshot.sides ?? {}) as Record<string, SideState>;
}

export const epoch: GameView = {
  Board: EpochBoard,
  genre: "Lane battle through the ages",
  cardMap: "meadow",
  accent: "#e4b440", // the gold trim in the sprites
  liveMap: "meadow",
  look,
  stats(_board, snapshot) {
    const result: Record<string, SeatStat[]> = {};
    for (const [seat, [, hp, gold, xp]] of Object.entries(sides(snapshot))) {
      // A destroyed base has no numbers, so the scoreboard says it is knocked out.
      if (hp > 0) result[seat] = [{ label: "HP", value: hp }, { label: "gold", value: gold }, { label: "XP", value: xp }];
    }
    return result;
  },
  strength(rawBoard, snapshot) {
    const board = rawBoard as EpochBoardData;
    const army: Record<string, number> = Object.fromEntries(Object.keys(sides(snapshot)).map((seat) => [seat, 0]));
    for (const [seat, kind] of Object.values((snapshot.units ?? {}) as Record<string, [string, string]>)) army[seat] += board.units[kind].price;
    return army;
  },
  strengthLabel: "Army (gold)",
  play: {
    Controls: EpochControls,
    viewBox: (board) => [0, 0, (board as EpochBoardData).length, HEIGHT],
    help: [
      "Train units with the buttons over the sky or the keys 1, 2 and 3: melee, ranged and heavy. Melee beats ranged, ranged beats heavy and heavy beats melee.",
      "T builds a turret on your base, the second and third opening a new place first. S sells the oldest for half its price.",
      "Kills earn gold, and kills your units help with earn XP. A advances to the next age once you have the XP and the gold: new units, a new turret and a stronger base.",
      "Destroy the enemy base to win. After eight minutes the base with more hit points left wins.",
    ],
    map: "meadow",
  },
};
