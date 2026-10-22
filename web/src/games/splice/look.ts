// Splice's pages look like its default Petri Dish theme: pale agar, dark green-grey ink
// and the magenta stain of the first player (games/splice/themes/petri_dish.json),
// under a rounded heading font as soft as the cells.

import { Quicksand } from "next/font/google";

import type { GameLook } from "../types.ts";

const quicksand = Quicksand({ subsets: ["latin"], weight: "700", variable: "--font-splice", preload: false });

export const look: GameLook = {
  scheme: "light",
  colors: {
    bg: "#e3eadb",
    surface: "#f3f6ec",
    "surface-2": "#dae3d0",
    text: "#1d2b28",
    muted: "#55665f",
    border: "#c7d3bd",
    "border-strong": "#a3b49a",
    accent: "#c2307a",
    "accent-text": "#9a1f5e",
    "accent-soft": "rgb(194 48 122 / 0.12)",
    "on-accent": "#fff6fa",
    "stage-accent": "#11808c",
    win: "#c9e2bd",
    loss: "#efc6d6",
  },
  font: { className: quicksand.variable, family: "var(--font-splice), system-ui, sans-serif" },
  backdrop: "/games/splice/agar.svg",
};
