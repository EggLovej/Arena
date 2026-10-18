// The games the website can draw. Adding a game means adding its folder here
// with a board and its scoreboard numbers; every page around them is shared.
// The game list shows them in this order, and the first is the home page's.

import type { GameData, MapEntry } from "@/lib/types.ts";

import type { GameView } from "./types.ts";
import { graphite } from "./graphite/index.ts";
import { ricochet } from "./ricochet/index.ts";

export const GAME_VIEWS: Record<string, GameView> = {
  graphite,
  ricochet,
};

export function gameView(id: string): GameView {
  const view = GAME_VIEWS[id];
  if (!view) throw new Error(`The website has no board for game ${id}; add one in src/games/`);
  return view;
}

/** The map that stands for the game: beside its name in the game list, and in its theme picker. */
export function cardMap(game: GameData): MapEntry | undefined {
  return game.maps.find((entry) => entry.id === gameView(game.id).cardMap) ?? game.maps[0];
}
