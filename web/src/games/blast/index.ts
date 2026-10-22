import type { Snapshot } from "@/lib/types.ts";

import type { GameView, SeatStat } from "../types.ts";
import { BlastBoard, viewBox, type BlastBoardData, type BomberState } from "./BlastBoard.tsx";
import { BlastControls } from "./BlastControls.tsx";
import { look } from "./look.ts";

function bombers(snapshot: Snapshot): Record<string, BomberState> {
  return (snapshot.bombers ?? {}) as Record<string, BomberState>;
}

export const blast: GameView = {
  Board: BlastBoard,
  genre: "Bombs on a grid, up to four",
  cardMap: "courtyard",
  accent: "#ff8a1c", // the flames' orange
  liveMap: "courtyard", // four bombers at once
  look,
  botIcons: ["digger_ai", "hermit_ai", "hunter_ai", "rookie_ai", "trapper_ai"],
  stats(_board, snapshot) {
    const result: Record<string, SeatStat[]> = {};
    for (const [seat, [, , , , , , bombs, range, speed, out]] of Object.entries(bombers(snapshot))) {
      // A caught bomber has no numbers, so the scoreboard says it is knocked out.
      if (out < 0) result[seat] = [{ label: bombs === 1 ? "bomb" : "bombs", value: bombs }, { label: "range", value: range }, { label: "speed", value: speed + 1 }];
    }
    return result;
  },
  strength(_board, snapshot) {
    return Object.fromEntries(
      Object.entries(bombers(snapshot)).map(([seat, [, , , , , , bombs, range, speed, out]]) => [
        seat,
        out < 0 ? bombs + range + speed : 0,
      ]),
    );
  },
  strengthLabel: "Bombs, range and speed added up (none once caught)",
  play: {
    Controls: BlastControls,
    viewBox: (board) => viewBox(board as BlastBoardData),
    help: [
      "The arrow keys or W A S D walk, and Space or the Bomb button drops a bomb where you stand. Hold two arrows to take a corner: you go the way pressed last wherever it is open, and the other way until then. With a mouse or a finger, hold the board where you want to go, and press your bomber to drop a bomb.",
      "A bomb goes off after two and a half seconds in a cross that stops at walls and breaks the first crate it reaches. Its flames set off any bomb they reach. Bombs and crates block the way, and so do the other bombers.",
      "Some crates hide a power-up: one more bomb at a time, a blast one tile longer, or quicker steps. Walk over it to pick it up; a blast burns it.",
      "Anyone the flames reach is out. After a minute and a half the walls close in from the edge, a tile at a time. The last one standing wins.",
    ],
    map: "yard",
    pauseKey: "p", // Space drops a bomb
  },
};
