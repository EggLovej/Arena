// Blast's pages look like a bomb about to go off: warm charcoal round the bright
// grid, a faint pattern of floor tiles, the flames' orange, and a blocky arcade
// heading font.

import { Bungee } from "next/font/google";

import type { GameLook } from "../types.ts";

const bungee = Bungee({ subsets: ["latin"], weight: "400", variable: "--font-blast", preload: false });

export const look: GameLook = {
  scheme: "dark",
  colors: {
    bg: "#15120e",
    surface: "#1f1a14",
    "surface-2": "#2a231b",
    text: "#f6eee2",
    muted: "#b3a895",
    border: "#342b21",
    "border-strong": "#54473a",
    accent: "#ff8a1c",
    "accent-text": "#ffa552",
    "accent-soft": "rgb(255 138 28 / 0.15)",
    "on-accent": "#1b120a",
    "stage-accent": "#ff8a1c",
    win: "#1f4527",
    loss: "#552019",
  },
  font: { className: bungee.variable, family: "var(--font-blast), system-ui, sans-serif" },
  backdrop: "/games/blast/tiles.svg",
};
