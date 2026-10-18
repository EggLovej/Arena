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
      "Train units with the buttons over the sky or the keys 1 to 4: melee, ranged, heavy and, in the Future, the super soldier. They march in single file, and ranged units shoot past the one in front.",
      "Q, W and E build one of the age's three turrets in a free place, and R opens another place. S sells the cheapest turret for half its price.",
      "Kills earn gold, and both sides earn XP from them. A advances to the next age once you have the XP: new units and turrets, a new special and a stronger base.",
      "X calls your age's special, free but slow to recharge. Destroy the enemy base to win; when time runs out, the stronger base wins.",
    ],
    map: "meadow",
    music: { file: "/games/epoch/music.ogg", seconds: 30 },
  },
};
