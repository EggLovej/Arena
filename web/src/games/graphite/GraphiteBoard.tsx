// Graphite's board in the browser, drawn the way the game's Pygame renderer
// (games/graphite/renderer.py) draws it: paper, inked routes with travel times,
// bases sized by production, and fleets as arrowheads moving along their routes.
// The paper, ink and factions' colours are the theme's (games/graphite/themes).

import { useId } from "react";

import type { BoardProps } from "../types.ts";

export type GraphiteBoardData = {
  name: string;
  width: number;
  height: number;
  bases: { id: string; x: number; y: number; production: number }[];
  edges: { a: string; b: string; time: number }[];
};

type Theme = {
  colors: Record<"ink" | "paper" | "paper_light" | "paper_dark" | "muted" | "highlight" | "white", string> & { rule?: string };
  players: Record<string, { fill: string; dark: string }>;
  neutral: { fill: string };
  texture?: string; // a picture of the paper, laid over its colour
  grid?: number; // squared paper: the side of a square in map units, ruled in colors.rule, every fifth line bolder
};

// [owner seat or "" for neutral, whole units]
export type BaseState = [string, number];
// [owner seat, units, from base, to base, tick it left, travel time in seconds]
export type FleetState = [string, number, string, string, number, number];

const FONT = '"DejaVu Sans Condensed", "Arial Narrow", "Roboto Condensed", system-ui, sans-serif';

/** Maps with this many bases draw small bases with only their units on them (games/graphite/sizing.py). */
export const LARGE_MAP = 80;

/** Smallest and largest radius; crowded maps use smaller bases (games/graphite/sizing.py). */
function radiusRange(count: number): [number, number] {
  if (count >= 25) return [17, 29];
  if (count >= 15) return [20, 34];
  return [22, 40];
}

export function baseRadii(board: GraphiteBoardData): Record<string, number> {
  if (board.bases.length >= LARGE_MAP) {
    // Each base by its own rate, so one outsized prize doesn't shrink all the others.
    return Object.fromEntries(board.bases.map((base) => [base.id, Math.round(8 + 6 * Math.sqrt(base.production))]));
  }
  const [smallest, largest] = radiusRange(board.bases.length);
  const most = Math.max(0, ...board.bases.map((base) => base.production));
  return Object.fromEntries(
    board.bases.map((base) => {
      const share = most > 0 ? base.production / most : 0.5;
      return [base.id, Math.round(smallest + (largest - smallest) * share)];
    }),
  );
}

/** The part of the map that holds bases, with room for their labels. */
export function viewBox(board: GraphiteBoardData, compact: boolean): [number, number, number, number] {
  const radii = baseRadii(board);
  // Room for the production badge and name under each base, which large maps leave out.
  const labels = !compact && board.bases.length < LARGE_MAP;
  const margin = compact ? 12 : 30;
  let left = Infinity, top = Infinity, right = -Infinity, bottom = -Infinity;
  for (const base of board.bases) {
    const r = radii[base.id];
    left = Math.min(left, base.x - r - margin - (labels ? 30 : 0));
    right = Math.max(right, base.x + r + margin + (labels ? 30 : 0));
    top = Math.min(top, base.y - r - margin);
    bottom = Math.max(bottom, base.y + r + margin + (labels ? 42 : 0));
  }
  left = Math.max(0, left);
  top = Math.max(0, top);
  right = Math.min(board.width, right);
  bottom = Math.min(board.height, bottom);
  return [left, top, right - left, bottom - top];
}

/**
 * The map a quarter turn anticlockwise, so what was on the left (Red's side on two-player maps)
 * is at the bottom. Only the bases move: names and numbers stay upright.
 */
export function turned(board: GraphiteBoardData): GraphiteBoardData {
  return { ...board, width: board.height, height: board.width, bases: board.bases.map((base) => ({ ...base, x: base.y, y: board.width - base.x })) };
}

function textWidth(text: string, size: number): number {
  return text.length * size * 0.56;
}

export function formatRate(rate: number): string {
  return rate > 0 ? `+${Number(rate.toPrecision(6))}/s` : "0/s";
}

export function GraphiteBoard({ board: rawBoard, snapshot, tick, tickSeconds, theme: rawTheme, compact = false }: BoardProps) {
  const board = rawBoard as GraphiteBoardData;
  const theme = rawTheme as Theme;
  const { ink, paper, paper_light: paperLight, paper_dark: paperDark, white, rule } = theme.colors;
  const { texture, grid } = theme;
  const squares = useId();
  const color = (owner: string) => (owner ? theme.players[owner]?.fill ?? theme.neutral.fill : theme.neutral.fill);

  const bases = (snapshot.bases ?? {}) as Record<string, BaseState>;
  const fleets = (snapshot.fleets ?? {}) as Record<string, FleetState>;
  const byId = Object.fromEntries(board.bases.map((base) => [base.id, base]));
  const radii = baseRadii(board);
  const dense = board.bases.length >= 25;
  const large = board.bases.length >= LARGE_MAP;
  const route = large ? [5, 2] : compact ? [9, 4] : [7, 3];
  const fleetScale = large ? 0.7 : 1;
  const nameSize = dense ? 12 : board.bases.length >= 15 ? 14 : 16;
  const productionSize = nameSize + 1;
  const [x0, y0, width, height] = viewBox(board, compact);

  return (
    <svg
      viewBox={`${x0} ${y0} ${width} ${height}`}
      role="img"
      aria-label={`${board.name} map`}
      // A preview fills its frame, with paper round the edges where the shapes differ.
      style={{ fontFamily: FONT, display: "block", width: "100%", height: compact ? "100%" : "auto", background: paper }}
    >
      {texture && (
        <image
          href={texture}
          x={0}
          y={0}
          width={board.width}
          height={board.height}
          preserveAspectRatio="xMidYMid slice"
          opacity={compact ? 0.6 : 1}
        />
      )}
      {grid && (
        <>
          <pattern id={squares} width={grid * 5} height={grid * 5} patternUnits="userSpaceOnUse">
            <path d={[1, 2, 3, 4].map((n) => `M ${n * grid} 0 V ${grid * 5} M 0 ${n * grid} H ${grid * 5}`).join(" ")} stroke={rule} strokeWidth={1} />
            <path d={`M 0 0 V ${grid * 5} M 0 0 H ${grid * 5}`} stroke={rule} strokeWidth={2.5} />
          </pattern>
          {/* Beyond the map too, where a preview's frame is a different shape. */}
          <rect x={x0 - width} y={y0 - height} width={width * 3} height={height * 3} fill={`url(#${squares})`} />
        </>
      )}

      {board.edges.map((edge) => {
        const a = byId[edge.a];
        const b = byId[edge.b];
        return (
          <g key={`${edge.a}|${edge.b}`}>
            <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={ink} strokeWidth={route[0]} />
            <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={paperDark} strokeWidth={route[1]} />
          </g>
        );
      })}

      {!compact && !dense &&
        board.edges.map((edge) => {
          const a = byId[edge.a];
          const b = byId[edge.b];
          const label = `${Number(edge.time.toPrecision(6))}s`;
          const w = textWidth(label, 14) + 14;
          const h = 22;
          const cx = (a.x + b.x) / 2;
          const cy = (a.y + b.y) / 2;
          return (
            <g key={`tag|${edge.a}|${edge.b}`}>
              <rect x={cx - w / 2} y={cy - h / 2} width={w} height={h} rx={5} fill={paperLight} stroke={ink} strokeWidth={2} />
              <text x={cx} y={cy} fontSize={14} fontWeight={700} fill={ink} textAnchor="middle" dominantBaseline="central">
                {label}
              </text>
            </g>
          );
        })}

      {Object.entries(fleets).map(([id, [owner, units, from, to, departed, total]]) => {
        const source = byId[from];
        const target = byId[to];
        const progress = Math.min(1, Math.max(0, ((tick - departed) * tickSeconds) / total));
        const x = source.x + (target.x - source.x) * progress;
        const y = source.y + (target.y - source.y) * progress;
        const angle = Math.atan2(target.y - source.y, target.x - source.x);
        const dx = Math.cos(angle);
        const dy = Math.sin(angle);
        const fill = color(owner);
        return (
          <g key={`fleet-${id}`}>
            <circle cx={x - dx * 27 * fleetScale} cy={y - dy * 27 * fleetScale} r={3 * fleetScale} fill={fill} />
            <circle cx={x - dx * 42 * fleetScale} cy={y - dy * 42 * fleetScale} r={2 * fleetScale} fill={fill} />
            <g transform={`translate(${x} ${y}) rotate(${(angle * 180) / Math.PI}) scale(${fleetScale})`}>
              <polygon points="17,0 -12,11 -8,0 -12,-11" fill={ink} />
              <polygon points="13.94,0 -9.84,9.02 -6.56,0 -9.84,-9.02" fill={fill} />
            </g>
            {!compact && (
              <text x={x - 2} y={y} fontSize={13 * fleetScale} fontWeight={700} fill={white} textAnchor="middle" dominantBaseline="central">
                {units}
              </text>
            )}
          </g>
        );
      })}

      {board.bases.map((base) => {
        const [owner, units] = bases[base.id] ?? ["", 0];
        const r = radii[base.id];
        const fill = color(owner);
        const textColor = owner ? white : ink;
        if (compact) {
          return (
            <g key={base.id}>
              <circle cx={base.x} cy={base.y} r={r + 3} fill={ink} />
              <circle cx={base.x} cy={base.y} r={r} fill={paperLight} />
              <circle cx={base.x} cy={base.y} r={r - 5} fill={owner ? fill : paperDark} />
            </g>
          );
        }
        const label = String(units);
        if (large) {
          // Too many bases for badges and names: the size shows production, and the units fit inside.
          let size = Math.round(r * 0.95);
          while (textWidth(label, size) > 2 * (r - 4) && size > 7) size -= 1;
          return (
            <g key={base.id}>
              <title>{`${base.id} · ${formatRate(base.production)}`}</title>
              <circle cx={base.x} cy={base.y} r={r + 2} fill={ink} />
              <circle cx={base.x} cy={base.y} r={r} fill={paperLight} />
              <circle cx={base.x} cy={base.y} r={r - 3} fill={owner ? fill : paperDark} />
              <text x={base.x} y={base.y} fontSize={size} fontWeight={700} fill={textColor} textAnchor="middle" dominantBaseline="central">
                {label}
              </text>
            </g>
          );
        }
        let size = Math.max(13, Math.round(r * 0.92));
        while (textWidth(label, size) > 2 * (r - 7) && size > 11) size -= 1;
        const rate = formatRate(base.production);
        const badgeWidth = textWidth(rate, productionSize) + 14;
        const badgeHeight = productionSize + 8;
        const badgeTop = base.y + r - badgeHeight / 2;
        const nameWidth = textWidth(base.id, nameSize) + (dense ? 10 : 16);
        const nameHeight = nameSize + 8;
        const nameTop = badgeTop + badgeHeight + 3;
        return (
          <g key={base.id}>
            <circle cx={base.x} cy={base.y} r={r + 3} fill={ink} />
            <circle cx={base.x} cy={base.y} r={r} fill={paperLight} />
            <circle cx={base.x} cy={base.y} r={r - 5} fill={owner ? fill : paperDark} />
            <text
              x={base.x}
              y={base.y - Math.floor(r / 8)}
              fontSize={size}
              fontWeight={700}
              fill={textColor}
              textAnchor="middle"
              dominantBaseline="central"
            >
              {label}
            </text>
            <rect
              x={base.x - badgeWidth / 2}
              y={badgeTop}
              width={badgeWidth}
              height={badgeHeight}
              rx={badgeHeight / 2}
              fill={owner ? fill : paperLight}
              stroke={ink}
              strokeWidth={2}
            />
            <text
              x={base.x}
              y={badgeTop + badgeHeight / 2}
              fontSize={productionSize}
              fontWeight={700}
              fill={owner ? white : ink}
              textAnchor="middle"
              dominantBaseline="central"
            >
              {rate}
            </text>
            <rect
              x={base.x - nameWidth / 2}
              y={nameTop}
              width={nameWidth}
              height={nameHeight}
              rx={3}
              fill={paperLight}
              stroke={ink}
              strokeWidth={2}
            />
            <text
              x={base.x}
              y={nameTop + nameHeight / 2}
              fontSize={nameSize}
              fontWeight={700}
              fill={ink}
              textAnchor="middle"
              dominantBaseline="central"
            >
              {base.id}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
