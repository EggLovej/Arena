// Splice's board in the browser: hexagon cells and the players' pieces drawn as living
// cells, in the theme's colours (games/splice/themes). The last move plays out once, in
// real time (SpliceBoard.css): a piece buds into its new cell or leaps there, then the
// pieces it won over change colour. So a paused or finished match shows it done.

import { useId, type ReactNode } from "react";

import type { BoardProps } from "../types.ts";
import "./SpliceBoard.css";

export type SpliceBoardData = { cells: [number, number][] }; // [column, row]: a row's cells are two columns apart

export type SpliceTheme = {
  glow: boolean; // pieces glow in the dark rather than sit on agar
  colors: Record<"background" | "rim" | "cell" | "line" | "ink" | "highlight", string>;
  players: Record<string, { fill: string; light: string; dark: string }>;
};

// [seat, from, to, tick, [[cell, previous owner] for each piece won], [cells filled at the end]]
export type LastMove = [string, number, number, number, [number, string][], number[]];

export const SIDE = 20; // a hexagon's side, and the distance from its centre to a corner
const WIDTH = Math.sqrt(3) * SIDE; // from one side of a hexagon to the other, and between neighbours in a row
const MARGIN = 10;
const BAND = 16; // above the cells, where whose turn it is shows
const PIECE = 0.72 * SIDE;

export function centre([column, row]: [number, number]): [number, number] {
  return [(column * WIDTH) / 2, row * 1.5 * SIDE];
}

/** Steps between two cells, as distance() in games/splice/splice/rules.py works them out. */
export function distance(a: [number, number], b: [number, number]): number {
  const columns = Math.abs(a[0] - b[0]);
  const rows = Math.abs(a[1] - b[1]);
  return rows + Math.max(0, (columns - rows) / 2);
}

/** The cells with a margin round them and, unless `compact`, the band above them that says whose turn it is. */
export function viewBox(board: SpliceBoardData, compact: boolean): [number, number, number, number] {
  const xs = board.cells.map((cell) => centre(cell)[0]);
  const ys = board.cells.map((cell) => centre(cell)[1]);
  const left = Math.min(...xs) - WIDTH / 2 - MARGIN;
  const top = Math.min(...ys) - SIDE - MARGIN - (compact ? 0 : BAND);
  return [left, top, Math.max(...xs) + WIDTH / 2 + MARGIN - left, Math.max(...ys) + SIDE + MARGIN - top];
}

export function hexagon([x, y]: [number, number], side = SIDE): string {
  return [0, 1, 2, 3, 4, 5].map((corner) => {
    const angle = ((60 * corner - 90) * Math.PI) / 180;
    return `${x + side * Math.cos(angle)},${y + side * Math.sin(angle)}`;
  }).join(" ");
}

type PieceProps = { at: [number, number]; seat: string; cell: number; theme: SpliceTheme; ids: string; opacity?: number };

/** One piece: a cell with its membrane and nucleus, the nucleus turned a different way in each cell.
 * `ids` names the board's gradients (PieceGradients). */
export function Piece({ at: [x, y], seat, cell, theme, ids, opacity = 1 }: PieceProps) {
  const { light, dark } = theme.players[seat];
  return (
    <g transform={`translate(${x} ${y})`} opacity={opacity}>
      {theme.glow && <circle r={PIECE * 1.45} fill={`url(#${ids}-halo-${seat})`} />}
      <circle r={PIECE} fill={`url(#${ids}-body-${seat})`} stroke={dark} strokeWidth={1.6} />
      <ellipse rx={PIECE * 0.34} ry={PIECE * 0.26} transform={`rotate(${(cell * 137) % 360}) translate(${PIECE * 0.18} 0)`} fill={theme.glow ? light : dark} opacity={0.75} />
    </g>
  );
}

/** The gradients every piece is painted with, named after `ids`. */
export function PieceGradients({ theme, ids }: { theme: SpliceTheme; ids: string }) {
  return (
    <defs>
      {Object.entries(theme.players).map(([seat, { fill, light, dark }]) => (
        <g key={seat}>
          <radialGradient id={`${ids}-body-${seat}`} cx="38%" cy="34%" r="70%">
            <stop offset="0" stopColor={light} />
            <stop offset="0.55" stopColor={fill} />
            <stop offset="1" stopColor={theme.glow ? fill : dark} stopOpacity={theme.glow ? 0.75 : 1} />
          </radialGradient>
          <radialGradient id={`${ids}-halo-${seat}`}>
            <stop offset="0.5" stopColor={fill} stopOpacity={0.55} />
            <stop offset="1" stopColor={fill} stopOpacity={0} />
          </radialGradient>
        </g>
      ))}
    </defs>
  );
}

export function SpliceBoard({ board: rawBoard, snapshot, theme: rawTheme, seats, compact = false }: BoardProps) {
  const board = rawBoard as SpliceBoardData;
  const theme = rawTheme as SpliceTheme;
  const ids = useId().replace(/:/g, "");
  const [x0, y0, width, height] = viewBox(board, compact);
  const pieces = (snapshot.pieces ?? {}) as Record<string, string>;
  const turn = (snapshot.turn as { seat?: string } | undefined)?.seat;
  const last = (snapshot.last as { move?: LastMove } | undefined)?.move;
  const name = (seat: string) => seats.find((entry) => entry.id === seat)?.name ?? seat;
  const won = new Map(last?.[4]);
  const filled = new Set(last?.[5]);

  return (
    <svg
      viewBox={`${x0} ${y0} ${width} ${height}`}
      role="img"
      aria-label={`Splice board${turn ? `, ${name(turn)} to move` : ""}`}
      // A preview fills its frame, with the background round the edges where the shapes differ.
      style={{ display: "block", width: "100%", height: compact ? "100%" : "auto", background: theme.colors.background }}
    >
      <PieceGradients theme={theme} ids={ids} />

      {/* The board's rim, then its cells over it. */}
      {board.cells.map((cell, index) => (
        <polygon key={`rim-${index}`} points={hexagon(centre(cell))} fill={theme.colors.rim} stroke={theme.colors.rim} strokeWidth={compact ? 6 : 9} strokeLinejoin="round" />
      ))}
      {board.cells.map((cell, index) => (
        <polygon key={`cell-${index}`} points={hexagon(centre(cell), SIDE - 0.8)} fill={theme.colors.cell} stroke={theme.colors.line} strokeWidth={1.2} />
      ))}

      {last && !compact && (
        <polygon points={hexagon(centre(board.cells[last[2]]), SIDE - 3)} fill="none" stroke={theme.colors.highlight} strokeWidth={1.5} opacity={0.35} />
      )}

      {Object.entries(pieces).map(([key, seat]) => {
        const cell = Number(key);
        const at = centre(board.cells[cell]);
        const piece = <Piece at={at} seat={seat} cell={cell} theme={theme} ids={ids} />;
        // A part of the last move, keyed by its tick so that it plays again for each move.
        const moving = (className: string, child: ReactNode, style?: object) => (
          <g key={`${key}-${last![3]}`} className={className} style={{ transformOrigin: `${at[0]}px ${at[1]}px`, ...style }}>
            {child}
          </g>
        );
        if (last && cell === last[2]) {
          // A grow buds out of the piece it came from; a jump leaps over what is between.
          const [x, y] = centre(board.cells[last[1]]);
          const jump = distance(board.cells[last[1]], board.cells[cell]) === 2;
          return moving(jump ? "splice-jump" : "splice-grow", piece, { "--dx": `${x - at[0]}px`, "--dy": `${y - at[1]}px` });
        }
        if (won.has(cell)) {
          return (
            <g key={key}>
              {piece}
              {moving("splice-won", <Piece at={at} seat={won.get(cell)!} cell={cell} theme={theme} ids={ids} />)}
            </g>
          );
        }
        if (filled.has(cell)) return moving("splice-fill", piece);
        return <g key={key}>{piece}</g>;
      })}

      {turn && !compact && (
        // Centred over the board: a dot, a gap and the words, at about 0.56 em a letter.
        <g transform={`translate(${x0 + width / 2 - (12 + `${name(turn)} to move`.length * 5) / 2} ${y0 + MARGIN + 2})`} fontSize={9} fontWeight={700} fill={theme.colors.ink}>
          <circle cx={4.5} r={4.5} fill={theme.players[turn]?.fill} stroke={theme.players[turn]?.dark} strokeWidth={1} />
          <text x={12} dominantBaseline="central">
            {name(turn)} to move
          </text>
        </g>
      )}
    </svg>
  );
}
