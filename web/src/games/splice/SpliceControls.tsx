"use client";

// Playing Splice with the mouse or a finger. On your turn the empty cells you can grow into
// are marked: click one and a piece grows there. To jump, click one of your pieces first,
// then an empty cell two steps from it. Pointing at a cell shows the move and the pieces it
// would win. With a keyboard, Tab moves between your pieces and the cells you can move to,
// and Enter acts as a click. The move goes to WebPerson in games/splice/arena_game.py and
// shows faintly until it is played, on the next tick.

import { useEffect, useId, useState, type KeyboardEvent as KeyPress, type MouseEvent } from "react";

import { typing } from "@/components/usePlayback.ts";

import type { PlayControlsProps } from "../types.ts";
import "./SpliceControls.css";
import { Piece, PieceGradients, SIDE, centre, distance, hexagon, type SpliceBoardData, type SpliceTheme } from "./SpliceBoard.tsx";

export function SpliceControls({ board: rawBoard, snapshot, theme: rawTheme, seat, person, active, send, layer, toBoard }: PlayControlsProps) {
  const board = rawBoard as SpliceBoardData;
  const theme = rawTheme as SpliceTheme;
  const ids = useId().replace(/:/g, "");
  const [selected, setSelected] = useState<number | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);

  const cells = board.cells;
  const pieces = (snapshot.pieces ?? {}) as Record<string, string>;
  const owner = (cell: number) => pieces[String(cell)];
  const pending = (person as { move?: [number, number] } | undefined)?.move ?? null;
  const ready = active && (snapshot.turn as { seat?: string } | undefined)?.seat === seat && !pending;
  const mine = cells.flatMap((_, cell) => (owner(cell) === seat ? [cell] : []));
  const steps = (a: number, b: number) => distance(cells[a], cells[b]);
  // A piece that has been won over cannot stay selected.
  const source = ready && selected !== null && owner(selected) === seat ? selected : null;

  /** Where a move to this empty cell would start: the selected piece if it reaches, else a piece next to
   * it (a grow, from whichever piece), else the one piece that can jump there; null if none or several. */
  const from = (target: number): number | null => {
    if (owner(target)) return null;
    if (source !== null && [1, 2].includes(steps(source, target))) return source;
    const near = mine.find((cell) => steps(cell, target) === 1);
    if (near !== undefined) return near;
    const far = mine.filter((cell) => steps(cell, target) === 2);
    return far.length === 1 ? far[0] : null;
  };
  const won = (target: number) => cells.flatMap((_, cell) => (owner(cell) && owner(cell) !== seat && steps(cell, target) === 1 ? [cell] : []));
  // The cells to mark: those the selected piece reaches, or else every cell a piece can grow into.
  const targets = !ready
    ? []
    : cells.flatMap((_, cell) => {
        if (owner(cell)) return [];
        const reach = source === null ? (mine.some((piece) => steps(piece, cell) === 1) ? 1 : 0) : steps(source, cell);
        return reach === 1 || reach === 2 ? [{ cell, jump: reach === 2 }] : [];
      });

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !typing(event)) setSelected(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const cellAt = (event: MouseEvent): number | null => {
    const point = toBoard(event);
    if (!point) return null;
    let best: number | null = null;
    let nearest = SIDE;
    cells.forEach((cell, index) => {
      const [x, y] = centre(cell);
      const gap = Math.hypot(point.x - x, point.y - y);
      if (gap < nearest) {
        best = index;
        nearest = gap;
      }
    });
    return best;
  };

  const choose = (cell: number | null) => {
    if (!ready || cell === null || (owner(cell) && owner(cell) !== seat)) {
      setSelected(null);
      return;
    }
    if (owner(cell) === seat) {
      setSelected(cell === source ? null : cell);
      return;
    }
    const start = from(cell);
    if (start !== null) send([start, cell]);
    setSelected(null);
  };

  const press = (cell: number) => (event: KeyPress) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    choose(cell);
  };

  const describe = (cell: number) => {
    if (owner(cell) === seat) return `Your piece ${cell}${cell === source ? ", selected" : ""}`;
    const start = from(cell);
    const gain = won(cell).length + (start !== null && steps(start, cell) === 1 ? 1 : 0);
    return `Empty cell ${cell}: ${start !== null && steps(start, cell) === 2 ? `jump from ${start}` : "grow"}, gaining ${gain}`;
  };

  // What pointing at a cell would do: the move drawn faintly, with the pieces it would win ringed.
  const preview = ready && hovered !== null ? from(hovered) : null;
  const shown: [number, number] | null = pending ?? (preview !== null && hovered !== null ? [preview, hovered] : null);
  const colour = theme.players[seat];

  return (
    <svg
      {...layer}
      style={{ cursor: ready && hovered !== null && (owner(hovered) === seat || preview !== null) ? "pointer" : "default" }}
      onClick={(event) => choose(cellAt(event))}
      onContextMenu={(event) => {
        event.preventDefault();
        setSelected(null);
      }}
      onMouseMove={(event) => setHovered(cellAt(event))}
      onMouseLeave={() => setHovered(null)}
      role="application"
      aria-label="The board: pick an empty cell to grow into, or one of your pieces and then a cell two steps away to jump"
    >
      <PieceGradients theme={theme} ids={ids} />
      {targets.map(({ cell, jump }) =>
        jump ? (
          <circle key={cell} className="splice-target" cx={centre(cells[cell])[0]} cy={centre(cells[cell])[1]} r={SIDE * 0.45} stroke={colour?.fill} />
        ) : (
          <circle key={cell} cx={centre(cells[cell])[0]} cy={centre(cells[cell])[1]} r={SIDE * 0.2} fill={colour?.fill} opacity={0.45} />
        ),
      )}
      {source !== null && <polygon points={hexagon(centre(cells[source]), SIDE - 2)} fill="none" stroke={theme.colors.highlight} strokeWidth={2.5} />}
      {shown && (
        <g pointerEvents="none">
          {won(shown[1]).map((cell) => (
            <circle key={cell} cx={centre(cells[cell])[0]} cy={centre(cells[cell])[1]} r={SIDE * 0.82} fill="none" stroke={colour?.fill} strokeWidth={2.5} />
          ))}
          <Piece at={centre(cells[shown[1]])} seat={seat} cell={shown[1]} theme={theme} ids={ids} opacity={0.55} />
          {steps(shown[0], shown[1]) === 2 && (
            <polygon points={hexagon(centre(cells[shown[0]]), SIDE - 2)} fill="none" stroke={theme.colors.highlight} strokeWidth={1.5} strokeDasharray="4 3" />
          )}
        </g>
      )}
      {ready &&
        cells.map((cell, index) =>
          owner(index) === seat || from(index) !== null ? (
            <polygon
              key={`key-${index}`}
              className="splice-key"
              points={hexagon(centre(cell), SIDE - 2)}
              stroke={theme.colors.highlight}
              tabIndex={0}
              role="button"
              aria-label={describe(index)}
              onKeyDown={press(index)}
              onMouseDown={(event) => event.preventDefault()} // a click does not take the keyboard's focus, so Space still pauses
            />
          ) : null,
        )}
    </svg>
  );
}
