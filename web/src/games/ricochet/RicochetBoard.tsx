// Ricochet's arena in the browser, drawn in the style of the chosen theme as the
// game's own window draws it (games/ricochet/src/ricochet/display/styles): the
// style draws the floor, walls, tanks and shots; every theme shows the shield,
// a hit's flash, health above each tank and energy below it the same way.

import { useId, useMemo, type ReactNode } from "react";

import type { BoardProps } from "../types.ts";
import { neon } from "./neon.tsx";
import { robotLeague } from "./robotLeague.tsx";

export type RicochetBoardData = {
  name: string;
  width: number;
  height: number;
  walls: [number, number, number, number][]; // x, y, width, height
  tank_radius: number;
  shot_radius: number;
  health: number;
  energy: number;
};

/** One of the game's themes (ricochet.themes): its colours, and the style that draws it. */
export type Theme = Record<"background" | "panel" | "text" | "muted" | "accent" | "energy" | "inactive" | "shield", string> & {
  style: string;
  players: Record<string, string>;
};

/** How a style draws. Tanks and shots are drawn round the origin, and tanks aim `aim` degrees from the x axis. */
export type Style = {
  /** The floor, its edge and the walls: what never moves. `id` makes the ids of its patterns unique to the board. */
  arena: (board: RicochetBoardData, theme: Theme, compact: boolean, id: string) => ReactNode;
  tank: (color: string, theme: Theme, radius: number, aim: number) => ReactNode;
  wreck: (color: string, theme: Theme, radius: number, aim: number) => ReactNode;
  shot: (color: string, theme: Theme, radius: number, bounced: boolean) => ReactNode;
};

const STYLES: Record<string, Style> = { neon, "robot-league": robotLeague };

// [x, y, aim in degrees]
export type TankState = [number, number, number];
// [health, energy, shield up (1 or 0), tick last hit (-1 if never)]
export type StatusState = [number, number, number, number];
// [seat, x, y, x speed, y speed (pixels a second), tick it was at x and y, bounced (1 or 0)]
type ShotState = [string, number, number, number, number, number, number];

const TRAIL_SECONDS = 0.07;
const FLASH_TICKS = 9; // how long a tank glows white after a hit

function shotAt(shot: ShotState, tick: number, tickSeconds: number): [number, number] {
  const [, x, y, vx, vy, from] = shot;
  const seconds = Math.max(0, tick - from) * tickSeconds;
  return [x + vx * seconds, y + vy * seconds];
}

/** Where a tank is at a moment between two ticks, and where it aims. */
function tankAt(now: TankState, next: TankState | undefined, fraction: number): TankState {
  if (!next || fraction <= 0) return now;
  const turn = ((((next[2] - now[2]) % 360) + 540) % 360) - 180; // the short way round
  return [now[0] + (next[0] - now[0]) * fraction, now[1] + (next[1] - now[1]) * fraction, now[2] + turn * fraction];
}

export function RicochetBoard({ board: rawBoard, snapshot, next, tick, tickSeconds, theme: rawTheme, seats, compact = false }: BoardProps) {
  const board = rawBoard as RicochetBoardData;
  const theme = rawTheme as Theme;
  const style = STYLES[theme.style] ?? neon; // a style the site can't draw yet looks like the default
  const id = useId();
  // Drawn once rather than every frame of a match.
  const arena = useMemo(() => style.arena(board, theme, compact, id), [style, board, theme, compact, id]);
  const { width, height, tank_radius: radius, shot_radius: shotRadius } = board;
  const tanks = (snapshot.tanks ?? {}) as Record<string, TankState>;
  const nextTanks = (next?.tanks ?? {}) as Record<string, TankState>;
  const status = (snapshot.status ?? {}) as Record<string, StatusState>;
  const shots = (snapshot.shots ?? {}) as Record<string, ShotState>;
  const whole = Math.floor(tick);
  const fraction = tick - whole;
  const color = (seat: string) => theme.players[seat] ?? seats.find((entry) => entry.id === seat)?.color ?? theme.text;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${board.name} arena`}
      // A preview fills its frame, the theme's background round the edges where the shapes differ.
      style={{ display: "block", width: "100%", height: compact ? "100%" : "auto", background: theme.background }}
    >
      {arena}

      {!compact &&
        Object.entries(shots).map(([key, shot]) => {
          const [x, y] = shotAt(shot, tick, tickSeconds);
          const [, x0, y0, vx, vy] = shot;
          // The trail reaches back along the flight, but not past where the shot set off or bounced.
          const tailX = Math.abs(vx * TRAIL_SECONDS) > Math.abs(x - x0) ? x0 : x - vx * TRAIL_SECONDS;
          const tailY = Math.abs(vy * TRAIL_SECONDS) > Math.abs(y - y0) ? y0 : y - vy * TRAIL_SECONDS;
          const fill = color(shot[0]);
          return (
            <g key={`shot-${key}`}>
              <line x1={tailX} y1={tailY} x2={x} y2={y} stroke={fill} strokeWidth={3} strokeLinecap="round" opacity={0.45} />
              <g transform={`translate(${x} ${y})`}>{style.shot(fill, theme, shotRadius, shot[6] === 1)}</g>
            </g>
          );
        })}

      {Object.entries(tanks).map(([seat, state]) => {
        const [x, y, aim] = tankAt(state, nextTanks[seat], fraction);
        const [health = 1, energy = 0, shield = 0, hit = -1] = status[seat] ?? [];
        const fill = color(seat);
        if (compact) {
          return (
            <g key={seat}>
              <circle cx={x} cy={y} r={radius * 1.6} fill={fill} opacity={0.25} />
              <circle cx={x} cy={y} r={radius} fill={theme.background} stroke={fill} strokeWidth={6} />
              <circle cx={x} cy={y} r={radius * 0.35} fill={fill} />
            </g>
          );
        }
        if (health <= 0) {
          return (
            <g key={seat} transform={`translate(${x} ${y})`}>
              {style.wreck(fill, theme, radius, aim)}
            </g>
          );
        }
        const flash = hit >= 0 && whole - hit < FLASH_TICKS ? 1 - (tick - hit) / FLASH_TICKS : 0;
        return (
          <g key={seat} transform={`translate(${x} ${y})`}>
            {shield ? <circle r={radius + 9} fill={theme.shield} fillOpacity={0.14} stroke={theme.shield} strokeWidth={2.5} /> : null}
            {style.tank(fill, theme, radius, aim)}
            {flash > 0 && <circle r={radius + 2} fill={theme.text} opacity={0.75 * flash} />}
            <rect x={-radius} y={radius + 7} width={radius * 2} height={4} rx={2} fill={theme.muted} opacity={0.35} />
            <rect x={-radius} y={radius + 7} width={(radius * 2 * Math.min(energy, board.energy)) / board.energy} height={4} rx={2} fill={theme.energy} />
            {Array.from({ length: board.health }, (_, index) => (
              <circle
                key={index}
                cx={(index - (board.health - 1) / 2) * 9}
                cy={-radius - 11}
                r={3}
                fill={index < health ? fill : "none"}
                stroke={fill}
                strokeWidth={1.2}
              />
            ))}
          </g>
        );
      })}
    </svg>
  );
}
