// Which of a game's themes its boards are drawn in: the one the person chose for
// that game, remembered on this device, or else the game's first (its default).
// Components read it with useTheme (components/ThemedBoard.tsx).

import type { Theme } from "./types.ts";

const KEY = "arena-board-theme-"; // and the game's id

const thisVisit: Record<string, string> = {}; // choices made since the page loaded, kept even where storage is refused
const listeners = new Set<() => void>();

/** Calls `listener` whenever a choice is made, until the returned function is called. */
export function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** The id of the theme chosen for the game, "" if none. */
export function chosen(gameId: string): string {
  if (gameId in thisVisit) return thisVisit[gameId];
  try {
    return localStorage.getItem(KEY + gameId) ?? "";
  } catch {
    return "";
  }
}

/** Draws every board of the game in this theme from now on. Choosing the default forgets the choice. */
export function chooseTheme(gameId: string, themes: Theme[], id: string) {
  thisVisit[gameId] = id;
  try {
    if (id === themes[0].id) localStorage.removeItem(KEY + gameId);
    else localStorage.setItem(KEY + gameId, id);
  } catch {
    // Not remembered, but still used for this visit.
  }
  for (const listener of listeners) listener();
}

/** Runs before the first paint. Boards in a page as it first loads are drawn in the default theme
 * and marked until the browser knows the choice (ThemedBoard); this keeps the marked boards of each
 * game with a theme chosen out of sight meanwhile, so they don't flash the default. */
export const BOARD_THEME_SCRIPT = `try{var s="";for(var i=0;i<localStorage.length;i++){var k=localStorage.key(i);if(k.indexOf("${KEY}")===0)s+='[data-pending-theme="'+k.slice(${KEY.length})+'"]{visibility:hidden}'}if(s){var e=document.createElement("style");e.textContent=s;document.head.appendChild(e)}}catch(e){}`;
