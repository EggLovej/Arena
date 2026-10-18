// Epoch's pages look like its lane: a pale sky, cream panels, the art's dark brown
// ink and the gold of its trim, under a chunky heading font that suits the sprites.

import { Lilita_One } from "next/font/google";

import type { GameLook } from "../types.ts";

const lilita = Lilita_One({ subsets: ["latin"], weight: "400", variable: "--font-epoch", preload: false });

export const look: GameLook = {
  scheme: "light",
  colors: {
    bg: "#e4f0f1",
    surface: "#fbf8f1",
    "surface-2": "#efe5cf",
    text: "#2b211c",
    muted: "#5d6566",
    border: "#cbdadb",
    "border-strong": "#a3b8b9",
    accent: "#d6a032",
    "accent-text": "#8a5d0c",
    "accent-soft": "rgb(214 160 50 / 0.16)",
    "on-accent": "#2b211c",
    "stage-accent": "#6fa9bd",
    win: "#d3e5b7",
    loss: "#f1c9ba",
  },
  font: { className: lilita.variable, family: "var(--font-epoch), system-ui, sans-serif" },
};
