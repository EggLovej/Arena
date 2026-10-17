// Graphite's pages look like its board: warm field-notes paper, ink and the
// red of the brush-stroke factions (the paper_atlas theme in games/graphite).

import { Fraunces } from "next/font/google";

import type { GameLook } from "../types.ts";

const fraunces = Fraunces({ subsets: ["latin"], variable: "--font-graphite", preload: false });

export const look: GameLook = {
  scheme: "light",
  colors: {
    bg: "#efe7d6",
    surface: "#f9f3e6",
    "surface-2": "#e6dbc4",
    text: "#2a2825",
    muted: "#665d50",
    border: "#d8cbb0",
    "border-strong": "#b8a888",
    accent: "#bb3933",
    "accent-text": "#962c27",
    "accent-soft": "rgb(187 57 51 / 0.12)",
    "on-accent": "#fffaf0",
    "stage-accent": "#2a2825",
    win: "#c6dbb4",
    loss: "#eac0b4",
  },
  font: { className: fraunces.variable, family: "var(--font-graphite), Georgia, serif" },
  backdrop: "/games/graphite/paper.jpg",
};
