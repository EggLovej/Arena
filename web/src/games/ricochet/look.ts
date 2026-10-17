// Ricochet's pages look like its default Neon Circuit theme: a midnight court
// under a faint grid, violet edges, and cyan and coral for the two tanks
// (games/ricochet/src/ricochet/themes).

import { Chakra_Petch } from "next/font/google";

import type { GameLook } from "../types.ts";

const chakraPetch = Chakra_Petch({ subsets: ["latin"], weight: ["600"], variable: "--font-ricochet", preload: false });

export const look: GameLook = {
  scheme: "dark",
  colors: {
    bg: "#09081a",
    surface: "#100e26",
    "surface-2": "#18153a",
    text: "#eceaff",
    muted: "#948db9",
    border: "#262048",
    "border-strong": "#3d3370",
    accent: "#a657ff",
    "accent-text": "#c39bff",
    "accent-soft": "rgb(166 87 255 / 0.16)",
    "on-accent": "#0b0820",
    "stage-accent": "#a657ff",
    win: "#123f47",
    loss: "#4a1c33",
  },
  font: { className: chakraPetch.variable, family: "var(--font-ricochet), ui-monospace, monospace" },
  backdrop: "/games/ricochet/grid.svg",
};
