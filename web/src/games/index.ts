// The games the website can draw. Adding a game means adding its folder here
// with a board and its scoreboard numbers; every page around them is shared.

import type { GameView } from "./types.ts";
import { graphite } from "./graphite/index.ts";

export const GAME_VIEWS: Record<string, GameView> = {
  graphite,
};

export function gameView(id: string): GameView {
  const view = GAME_VIEWS[id];
  if (!view) throw new Error(`The website has no board for game ${id}; add one in src/games/`);
  return view;
}
