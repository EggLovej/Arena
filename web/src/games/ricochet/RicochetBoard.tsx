// Ricochet's arena in the browser, in its default Neon Circuit look (the
// "neon" style in games/ricochet/src/ricochet/display/styles): a midnight grid,
// violet-edged walls, ringed tanks with their barrels, health above and energy
// below, glowing shots and trails.

import type { BoardProps } from "../types.ts";

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

type Theme = Record<"background" | "panel" | "text" | "muted" | "accent" | "energy" | "inactive" | "shield", string> & {
  players: Record<string, string>;
};

// [x, y, aim in degrees]
export type TankState = [number, number, number];
// [health, energy, shield up (1 or 0), tick last hit (-1 if never)]
export type StatusState = [number, number, number, number];
// [seat, x, y, x speed, y speed (pixels a second), tick it was at x and y, bounced (1 or 0)]
type ShotState = [string, number, number, number, number, number, number];

const GRID = 40;
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
  const { width, height, tank_radius: radius, shot_radius: shotRadius } = board;
  const tanks = (snapshot.tanks ?? {}) as Record<string, TankState>;
  const nextTanks = (next?.tanks ?? {}) as Record<string, TankState>;
  const status = (snapshot.status ?? {}) as Record<string, StatusState>;
  const shots = (snapshot.shots ?? {}) as Record<string, ShotState>;
  const whole = Math.floor(tick);
  const fraction = tick - whole;
  const color = (seat: string) => theme.players[seat] ?? seats.find((entry) => entry.id === seat)?.color ?? theme.text;
  const id = compact ? "ricochet-compact" : "ricochet";

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${board.name} arena`}
      // A preview fills its frame, the arena's floor colour round the edges where the shapes differ.
      style={{ display: "block", width: "100%", height: compact ? "100%" : "auto", background: theme.background }}
    >
      <defs>
        <pattern id={`${id}-grid`} width={GRID} height={GRID} patternUnits="userSpaceOnUse">
          <path d={`M ${GRID} 0 L 0 0 0 ${GRID}`} fill="none" stroke={theme.inactive} strokeWidth={1} opacity={0.55} />
        </pattern>
        <radialGradient id={`${id}-floor`} cx="50%" cy="50%" r="70%">
          <stop offset="0%" stopColor={theme.accent} stopOpacity={0.08} />
          <stop offset="100%" stopColor={theme.accent} stopOpacity={0} />
        </radialGradient>
      </defs>

      <rect width={width} height={height} fill={theme.background} />
      <rect width={width} height={height} fill={`url(#${id}-grid)`} />
      <rect width={width} height={height} fill={`url(#${id}-floor)`} />

      {/* The arena's edge, which shots bounce off like any wall. */}
      <rect x={3} y={3} width={width - 6} height={height - 6} fill="none" stroke={theme.accent} strokeWidth={10} opacity={0.18} />
      <rect x={2} y={2} width={width - 4} height={height - 4} fill="none" stroke={theme.accent} strokeWidth={compact ? 6 : 3} />

      {board.walls.map(([x, y, w, h], index) => (
        <g key={index}>
          <rect x={x - 3} y={y - 3} width={w + 6} height={h + 6} rx={3} fill="none" stroke={theme.accent} strokeWidth={6} opacity={0.22} />
          <rect x={x} y={y} width={w} height={h} fill={theme.panel} stroke={theme.accent} strokeWidth={compact ? 5 : 2.5} />
          {!compact && w > 24 && h > 24 && (
            <rect x={x + 6} y={y + 6} width={w - 12} height={h - 12} fill="none" stroke={theme.accent} strokeWidth={1} opacity={0.35} />
          )}
        </g>
      ))}

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
              <circle cx={x} cy={y} r={shotRadius * 2.4} fill={fill} opacity={0.22} />
              <circle cx={x} cy={y} r={shotRadius} fill={shot[6] ? theme.text : fill} stroke={fill} strokeWidth={1.5} />
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
            <g key={seat} transform={`translate(${x} ${y})`} opacity={0.75}>
              <circle r={radius} fill={theme.panel} stroke={theme.inactive} strokeWidth={3} />
              <path d={`M ${-radius * 0.55} ${-radius * 0.55} L ${radius * 0.55} ${radius * 0.55} M ${radius * 0.55} ${-radius * 0.55} L ${-radius * 0.55} ${radius * 0.55}`} stroke={fill} strokeWidth={3} strokeLinecap="round" />
            </g>
          );
        }
        const flash = hit >= 0 && whole - hit < FLASH_TICKS ? 1 - (tick - hit) / FLASH_TICKS : 0;
        return (
          <g key={seat} transform={`translate(${x} ${y})`}>
            <circle r={radius * 1.7} fill={fill} opacity={0.14} />
            {shield ? <circle r={radius + 9} fill={theme.shield} fillOpacity={0.14} stroke={theme.shield} strokeWidth={2.5} /> : null}
            <g transform={`rotate(${aim})`}>
              <rect x={radius * 0.2} y={-3.5} width={radius * 1.5} height={7} rx={2} fill={fill} />
            </g>
            <circle r={radius} fill={theme.background} stroke={fill} strokeWidth={3} />
            <circle r={radius * 0.62} fill="none" stroke={fill} strokeWidth={2.5} strokeDasharray="7 4" />
            <circle r={radius * 0.28} fill={fill} />
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
