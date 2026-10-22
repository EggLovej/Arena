// Blast's grid in the browser: a floor between stone pillars, crates, bombs with
// burning fuses, the cross of every blast, the power-ups burned crates leave and
// the bombers, and from a minute and a half in the walls closing in from the edge.
// The colours are the theme's (games/blast/themes).

import { memo } from "react";

import type { BoardProps } from "../types.ts";

export type BlastBoardData = {
  name: string;
  rows: string[]; // "#" a wall, any other character floor
  fuse: number; // ticks from dropping a bomb to its blast
  closes: [number, number, number][]; // x, y and the tick each tile the walls close in on closes
};

type Colors = Record<
  | "floor"
  | "floor_alt"
  | "wall"
  | "wall_light"
  | "wall_dark"
  | "crate"
  | "crate_dark"
  | "ink"
  | "bomb"
  | "flame"
  | "flame_core"
  | "shadow"
  | "power"
  | "power_edge"
  | "speed",
  string
>;
type Theme = { colors: Colors; players: Record<string, { fill: string; dark: string }> };

// [x, y, the x and y its last step began at, the tick it began, the tick it can step again, bombs, range, speed, the tick it was caught or -1]
export type BomberState = [number, number, number, number, number, number, number, number, number, number];
// [x, y, owner, the tick it goes off]
type BombState = [number, number, string, number];
// [the tick it caught, the tick it burns out]
type FlameState = [number, number];

export const TILE = 48;
const WARN = 40; // ticks before a tile closes that it shows it is coming
const FALL = 6; // ticks a closing tile takes to drop into place
const FADE = 30; // ticks a caught bomber takes to fade away
const NEXT_TO: [number, number][] = [[0, -1], [0, 1], [-1, 0], [1, 0]];

export function viewBox(board: BlastBoardData): [number, number, number, number] {
  return [0, 0, board.rows[0].length * TILE, board.rows.length * TILE];
}

function tileOf(key: string): [number, number] {
  const [x, y] = key.split(",");
  return [Number(x), Number(y)];
}

/** A stone block, lit from the top left: the map's walls, and the tiles the walls close in on. */
const Block = memo(function Block({ x, y, colors }: { x: number; y: number; colors: Colors }) {
  const [left, top, right, bottom] = [x * TILE, y * TILE, (x + 1) * TILE, (y + 1) * TILE];
  const edge = 6;
  return (
    <g>
      <rect x={left} y={top} width={TILE} height={TILE} fill={colors.wall} />
      <path d={`M${left} ${top}H${right}l${-edge} ${edge}H${left + edge}V${bottom - edge}L${left} ${bottom}Z`} fill={colors.wall_light} />
      <path d={`M${right} ${top}V${bottom}H${left}l${edge} ${-edge}H${right - edge}V${top + edge}Z`} fill={colors.wall_dark} />
    </g>
  );
});

/** A crate: a frame round a braced panel, casting a shadow on the floor below it. */
const Crate = memo(function Crate({ x, y, colors }: { x: number; y: number; colors: Colors }) {
  const [left, top] = [x * TILE, y * TILE];
  return (
    <g>
      <rect x={left + 3} y={top + TILE - 2} width={TILE - 4} height={7} fill={colors.shadow} />
      <rect x={left + 3} y={top + 3} width={TILE - 6} height={TILE - 6} rx={4} fill={colors.crate} stroke={colors.crate_dark} strokeWidth={2.5} />
      <rect x={left + 10} y={top + 10} width={TILE - 20} height={TILE - 20} fill="none" stroke={colors.crate_dark} strokeWidth={2} />
      <path d={`M${left + 10} ${top + TILE - 10}L${left + TILE - 10} ${top + 10}`} stroke={colors.crate_dark} strokeWidth={3} />
    </g>
  );
});

/** The floor, checked, and the map's walls: what never changes in a match. */
const Ground = memo(function Ground({ board, colors, compact }: { board: BlastBoardData; colors: Colors; compact: boolean }) {
  const tiles = board.rows.flatMap((row, y) => [...row].map((cell, x) => ({ x, y, wall: cell === "#" })));
  return (
    <g>
      <rect width={board.rows[0].length * TILE} height={board.rows.length * TILE} fill={colors.floor} />
      {tiles.map(({ x, y, wall }) => (wall || (x + y) % 2 ? null : <rect key={`${x},${y}`} x={x * TILE} y={y * TILE} width={TILE} height={TILE} fill={colors.floor_alt} />))}
      {/* A pillar's shadow falls on the floor below it. */}
      {!compact &&
        tiles.map(({ x, y, wall }) =>
          wall && y + 1 < board.rows.length && board.rows[y + 1][x] !== "#" ? <rect key={`shadow ${x},${y}`} x={x * TILE} y={(y + 1) * TILE} width={TILE} height={7} fill={colors.shadow} /> : null,
        )}
      {tiles.map(({ x, y, wall }) => (wall ? <Block key={`${x},${y}`} x={x} y={y} colors={colors} /> : null))}
    </g>
  );
});

/** A power-up lying in the open: more bombs, a longer blast or quicker steps. */
function Power({ kind, colors }: { kind: string; colors: Colors }) {
  return (
    <g>
      <rect x={-15} y={-15} width={30} height={30} rx={8} fill={colors.power} stroke={colors.power_edge} strokeWidth={2.5} />
      {kind === "bomb" && (
        <g>
          <circle cx={-1} cy={2} r={8} fill={colors.bomb} />
          <path d="M4 -4 Q7 -9 10 -9" fill="none" stroke={colors.crate_dark} strokeWidth={2} />
          <circle cx={10} cy={-9} r={2.5} fill={colors.flame} />
        </g>
      )}
      {kind === "fire" && (
        <g>
          <path d="M0 -11 C7 -4 10 1 8 6 C6 11 -6 11 -8 6 C-10 1 -6 -2 -4 -7 C-3 -3 -1 -2 0 -11 Z" fill={colors.flame} />
          <path d="M0 -2 C4 2 5 5 3 8 C1 10 -3 10 -4 7 C-5 4 -2 2 0 -2 Z" fill={colors.flame_core} />
        </g>
      )}
      {kind === "speed" && <path d="M3 -12 L-7 2 H-1 L-4 12 L7 -3 H1 Z" fill={colors.speed} stroke={colors.ink} strokeWidth={1} strokeLinejoin="round" />}
    </g>
  );
}

/** A bomb, its fuse burning down and its cap in its owner's colour; it swells faster as the blast nears. */
function Bomb({ cap, left, fuse, tick, colors }: { cap: string; left: number; fuse: number; tick: number; colors: Colors }) {
  const swell = 1 + 0.07 * Math.sin(tick * (left < fuse / 3 ? 1.4 : 0.5));
  const burnt = 1 - Math.max(0, left) / fuse; // how much of the fuse has burned
  const [endX, endY] = [9 + 7 * (1 - burnt), -12 - 8 * (1 - burnt)];
  return (
    <g>
      <ellipse cx={0} cy={15} rx={13} ry={4} fill={colors.shadow} />
      <g transform={`scale(${swell})`}>
        <circle r={14} fill={colors.bomb} />
        <circle cx={-5} cy={-5} r={4} fill="#fff" opacity={0.35} />
        <path d={`M7 -10 Q${endX - 4} ${endY - 2} ${endX} ${endY}`} fill="none" stroke={colors.crate_dark} strokeWidth={2.5} />
        <rect x={3} y={-15} width={8} height={7} rx={2} transform="rotate(35 7 -11.5)" fill={cap} />
        <circle cx={endX} cy={endY} r={3 + Math.abs(Math.sin(tick * 2.1)) * 2} fill={colors.flame} />
        <circle cx={endX} cy={endY} r={1.8} fill={colors.flame_core} />
      </g>
    </g>
  );
}

/** Where a bomber stands at `tick` (part way through a step, if it is stepping), which way it faces, and how far through the step it is. */
function bomberAt(state: BomberState, tick: number): { x: number; y: number; dx: number; dy: number; through: number } {
  const [x, y, fromX, fromY, stepped, ready] = state;
  const through = ready > stepped ? Math.min(1, Math.max(0, (tick - stepped) / (ready - stepped))) : 1;
  const [dx, dy] = x === fromX && y === fromY ? [0, 1] : [Math.sign(x - fromX), Math.sign(y - fromY)];
  return { x: fromX + (x - fromX) * through, y: fromY + (y - fromY) * through, dx, dy, through };
}

/** A bomber: a round body in its seat's colour with a visor looking the way it goes, bobbing as it steps. */
function Bomber({ fill, dark, dx, dy, through, caught, colors }: { fill: string; dark: string; dx: number; dy: number; through: number; caught: boolean; colors: Colors }) {
  const bob = through < 1 ? -3 * Math.sin(through * Math.PI) : 0;
  const [eyeX, eyeY] = [dx * 3, dy * 2 - 1];
  return (
    <g>
      <ellipse cx={0} cy={18} rx={13} ry={4} fill={colors.shadow} />
      <g transform={`translate(0 ${bob})`}>
        <ellipse cx={-7} cy={14} rx={5.5} ry={3.5} fill={dark} />
        <ellipse cx={7} cy={14} rx={5.5} ry={3.5} fill={dark} />
        <line x1={0} y1={-14} x2={0} y2={-21} stroke={dark} strokeWidth={2.5} />
        <circle cy={-23} r={3.5} fill={fill} stroke={dark} strokeWidth={1.5} />
        <circle r={15} fill={fill} stroke={dark} strokeWidth={2.5} />
        <rect x={-10} y={-8} width={20} height={12} rx={6} fill="#fff" stroke={dark} strokeWidth={1.5} />
        {caught ? (
          <path d="M-6 -4 l4 4 m0 -4 l-4 4 M2 -4 l4 4 m0 -4 l-4 4" stroke={colors.ink} strokeWidth={1.6} strokeLinecap="round" />
        ) : (
          <g fill={colors.ink}>
            <circle cx={-4 + eyeX} cy={-2 + eyeY} r={2.2} />
            <circle cx={4 + eyeX} cy={-2 + eyeY} r={2.2} />
          </g>
        )}
      </g>
    </g>
  );
}

/** Every tile that burns as one path of rectangles `width` wide, reaching across to the burning tiles beside it so a blast reads as one cross. */
function flamePath(flames: Record<string, FlameState>, width: (flame: FlameState) => number): string {
  let path = "";
  for (const [key, flame] of Object.entries(flames)) {
    const [x, y] = tileOf(key);
    const half = width(flame) / 2;
    if (half <= 0) continue;
    const [cx, cy] = [(x + 0.5) * TILE, (y + 0.5) * TILE];
    path += `M${cx - half} ${cy - half}h${2 * half}v${2 * half}h${-2 * half}Z`;
    for (const [dx, dy] of NEXT_TO) {
      if (!(`${x + dx},${y + dy}` in flames)) continue;
      // Half a tile towards the neighbour, which reaches back the other half.
      const [w, h] = dx ? [TILE / 2, 2 * half] : [2 * half, TILE / 2];
      path += `M${dx < 0 ? cx - w : dx > 0 ? cx : cx - half} ${dy < 0 ? cy - h : dy > 0 ? cy : cy - half}h${w}v${h}h${-w}Z`;
    }
  }
  return path;
}

export function BlastBoard({ board: rawBoard, snapshot, tick, theme: rawTheme, seats, compact = false }: BoardProps) {
  const board = rawBoard as BlastBoardData;
  const { colors, players } = rawTheme as Theme;
  const [, , width, height] = viewBox(board);
  const player = (seat: string) => players[seat] ?? { fill: seats.find((entry) => entry.id === seat)?.color ?? colors.ink, dark: colors.ink };

  const bombers = (snapshot.bombers ?? {}) as Record<string, BomberState>;
  const crates = (snapshot.crates ?? {}) as Record<string, number>;
  const bombs = (snapshot.bombs ?? {}) as Record<string, BombState>;
  const flames = (snapshot.flames ?? {}) as Record<string, FlameState>;
  const powers = (snapshot.powers ?? {}) as Record<string, string>;
  const closed = (snapshot.closed ?? {}) as Record<string, number>;
  // How far a flame has burned, 0 when it caught and 1 when it is out: it flares, then dies down.
  const life = ([lit, out]: FlameState) => Math.min(1, Math.max(0, (tick - lit) / Math.max(1, out - lit)));

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${board.name} map`}
      // A preview fills its frame, stone round the edges where the shapes differ.
      style={{ display: "block", width: "100%", height: compact ? "100%" : "auto", background: colors.wall_dark }}
    >
      <Ground board={board} colors={colors} compact={compact} />

      {/* The next tiles the walls will close in on darken as their time comes. */}
      {board.closes.map(([x, y, closes]) =>
        closes > tick && closes <= tick + WARN ? (
          <rect key={`warn ${x},${y}`} x={x * TILE + 2} y={y * TILE + 2} width={TILE - 4} height={TILE - 4} rx={4} fill={colors.wall_dark} opacity={0.15 + 0.45 * (1 - (closes - tick) / WARN)} stroke={colors.flame} strokeWidth={2} strokeDasharray="6 5" />
        ) : null,
      )}

      {Object.entries(powers).map(([key, kind]) => {
        const [x, y] = tileOf(key);
        return (
          <g key={`power ${key}`} transform={`translate(${(x + 0.5) * TILE} ${(y + 0.5) * TILE + Math.sin(tick * 0.15 + x + y) * 1.5})`}>
            <Power kind={kind} colors={colors} />
          </g>
        );
      })}

      {Object.keys(crates).map((key) => {
        const [x, y] = tileOf(key);
        return <Crate key={`crate ${key}`} x={x} y={y} colors={colors} />;
      })}

      {Object.values(bombs).map(([x, y, owner, explodes]) => (
        <g key={`bomb ${x},${y}`} transform={`translate(${(x + 0.5) * TILE} ${(y + 0.5) * TILE})`}>
          <Bomb cap={player(owner).fill} left={explodes - tick} fuse={board.fuse} tick={tick} colors={colors} />
        </g>
      ))}

      <path d={flamePath(flames, (flame) => TILE * (0.92 - 0.5 * life(flame) ** 2))} fill={colors.flame} />
      <path d={flamePath(flames, (flame) => TILE * (0.5 - 0.42 * life(flame)))} fill={colors.flame_core} />

      {Object.entries(closed).map(([key, closes]) => {
        const [x, y] = tileOf(key);
        // A tile drops into place from above as it closes.
        const falling = Math.max(0, 1 - (tick - closes) / FALL);
        return (
          <g key={`closed ${key}`} transform={falling ? `translate(0 ${-falling * TILE * 0.6})` : undefined} opacity={1 - falling}>
            <Block x={x} y={y} colors={colors} />
          </g>
        );
      })}

      {Object.entries(bombers).map(([seat, state]) => {
        const out = state[9];
        if (out >= 0 && tick - out > FADE) return null;
        const { x, y, dx, dy, through } = bomberAt(state, tick);
        const gone = out >= 0 ? Math.max(0, (tick - out) / FADE) : 0;
        const { fill, dark } = player(seat);
        return (
          <g key={`bomber ${seat}`} transform={`translate(${(x + 0.5) * TILE} ${(y + 0.5) * TILE - gone * 12})`} opacity={1 - gone}>
            <Bomber fill={fill} dark={dark} dx={dx} dy={dy} through={through} caught={out >= 0} colors={colors} />
          </g>
        );
      })}
    </svg>
  );
}
