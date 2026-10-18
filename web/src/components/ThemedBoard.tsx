"use client";

// A game's board in the theme chosen for the game (lib/themes.ts), for boards that
// are part of a page as it first loads, such as map previews. Until the browser
// has read the choice, the board is in the default theme and marked, and
// BOARD_THEME_SCRIPT hides it if another theme was chosen. Boards drawn only
// later, such as a replay's, can take the theme from useTheme instead.

import { useSyncExternalStore } from "react";

import { gameView } from "@/games/index.ts";
import type { BoardProps } from "@/games/types.ts";
import { chosen, subscribe } from "@/lib/themes.ts";
import type { Theme } from "@/lib/types.ts";

type Props = Omit<BoardProps, "theme"> & { gameId: string; themes: Theme[] };

/** The theme to draw a game's boards in, and whether it is known: it isn't while a page drawn
 * on the server is being taken over by the browser, and the default stands in meanwhile. */
export function useTheme(gameId: string, themes: Theme[]): [Theme, boolean] {
  const choice = useSyncExternalStore(subscribe, () => chosen(gameId), () => null);
  return [themes.find((theme) => theme.id === choice) ?? themes[0], choice !== null];
}

export function ThemedBoard({ gameId, themes, ...props }: Props) {
  const { Board } = gameView(gameId);
  const [theme, known] = useTheme(gameId, themes);
  return (
    <div className="board" data-pending-theme={known ? undefined : gameId}>
      <Board {...props} theme={theme} />
    </div>
  );
}
