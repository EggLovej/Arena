"use client";

// Playing Graphite with the mouse or a finger, as in the game's own window
// (games/graphite/human_controller.py): click one of your bases, then a base next to
// it, and half its units go there. Shift-click the second base instead, or turn on
// Stream, and from then on everything that base has goes there, tick after tick, until
// you shift-click it again. The commands go to WebPerson in games/graphite/arena_game.py.

import { useEffect, useRef, useState, type MouseEvent } from "react";
import { createPortal } from "react-dom";

import type { PlayControlsProps } from "../types.ts";
import { baseRadii, viewBox, type GraphiteBoardData } from "./GraphiteBoard.tsx";

type BaseState = [string, number];
type Theme = { colors: Record<string, string>; players: Record<string, { fill: string; dark: string }> };

// How far outside a base's circle a click still counts, in map units: a finger is less precise than a mouse.
const MARGIN = { mouse: 6, touch: 16 };

export function GraphiteControls({ board: rawBoard, snapshot, theme: rawTheme, seat, person, active, send, tools }: PlayControlsProps) {
  const board = rawBoard as GraphiteBoardData;
  const theme = rawTheme as Theme;
  const svg = useRef<SVGSVGElement>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false); // the Stream switch, for screens without a Shift key
  const touch = useRef(false);

  const bases = (snapshot.bases ?? {}) as Record<string, BaseState>;
  const streams = ((person as { streams?: Record<string, string> } | undefined)?.streams ?? {}) as Record<string, string>;
  const radii = baseRadii(board);
  const byId = Object.fromEntries(board.bases.map((base) => [base.id, base]));
  const neighbours: Record<string, Set<string>> = {};
  for (const edge of board.edges) {
    (neighbours[edge.a] ??= new Set()).add(edge.b);
    (neighbours[edge.b] ??= new Set()).add(edge.a);
  }
  const mine = (id: string) => bases[id]?.[0] === seat;
  // A base that has been lost cannot stay selected.
  const source = selected && mine(selected) ? selected : null;
  const ink = theme.colors.ink ?? "#1c1b19";
  const colour = theme.players[seat]?.fill ?? "#3d5afe";

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSelected(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const baseAt = (event: MouseEvent): string | null => {
    const element = svg.current;
    const matrix = element?.getScreenCTM();
    if (!element || !matrix) return null;
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    const margin = touch.current ? MARGIN.touch : MARGIN.mouse;
    let best: string | null = null;
    let nearest = Infinity;
    for (const base of board.bases) {
      const distance = Math.hypot(point.x - base.x, point.y - base.y);
      if (distance <= radii[base.id] + margin && distance < nearest) {
        best = base.id;
        nearest = distance;
      }
    }
    return best;
  };

  const click = (event: MouseEvent) => {
    if (!active) return;
    const stream = event.shiftKey || streaming;
    const clicked = baseAt(event);
    if (clicked === null) {
      setSelected(null);
      return;
    }
    if (source === null) {
      if (stream && streams[clicked]) send({ stop: clicked });
      else if (mine(clicked)) setSelected(clicked);
      return;
    }
    const connected = neighbours[source]?.has(clicked);
    if (clicked === source) {
      if (stream && streams[source]) send({ stop: source });
      setSelected(null);
    } else if (stream && connected) {
      send({ stream: [source, clicked] });
      setSelected(null);
    } else if (connected && (bases[source]?.[1] ?? 0) > 0) {
      send({ send: [source, clicked] });
      setSelected(null);
    } else setSelected(mine(clicked) ? clicked : null);
  };

  const [x0, y0, width, height] = viewBox(board, false);
  const pointer = hovered && (source ? neighbours[source]?.has(hovered) || mine(hovered) : mine(hovered) || streams[hovered]);

  return (
    <>
      <svg
        ref={svg}
        viewBox={`${x0} ${y0} ${width} ${height}`}
        className="play-controls-layer"
        style={{ cursor: active && pointer ? "pointer" : "default" }}
        onPointerDown={(event) => (touch.current = event.pointerType !== "mouse")}
        onClick={click}
        onContextMenu={(event) => {
          event.preventDefault();
          setSelected(null);
        }}
        onMouseMove={(event) => setHovered(baseAt(event))}
        onMouseLeave={() => setHovered(null)}
        role="application"
        aria-label="The board: click one of your bases, then a base next to it, to send half its units"
      >
        <defs>
          <marker id="stream-head" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 Z" fill={colour} stroke={ink} strokeWidth="1.5" />
          </marker>
        </defs>
        {Object.entries(streams).map(([from, to]) => {
          const a = byId[from];
          const b = byId[to];
          if (!a || !b) return null;
          const length = Math.hypot(b.x - a.x, b.y - a.y);
          const start = radii[from] + 6;
          const end = length - radii[to] - 10;
          if (end <= start) return null;
          const ux = (b.x - a.x) / length;
          const uy = (b.y - a.y) / length;
          return (
            <line
              key={`stream-${from}`}
              className="play-stream"
              x1={a.x + ux * start}
              y1={a.y + uy * start}
              x2={a.x + ux * end}
              y2={a.y + uy * end}
              stroke={colour}
              strokeWidth={6}
              markerEnd="url(#stream-head)"
            />
          );
        })}
        {source &&
          [...(neighbours[source] ?? [])].map((id) => (
            <circle key={`target-${id}`} className="play-target" cx={byId[id].x} cy={byId[id].y} r={radii[id] + 7} stroke={colour} />
          ))}
        {hovered && active && hovered !== source && (source ? neighbours[source]?.has(hovered) : mine(hovered)) && (
          <circle cx={byId[hovered].x} cy={byId[hovered].y} r={radii[hovered] + 7} fill="none" stroke={ink} strokeWidth={3} opacity={0.5} />
        )}
        {source && (
          <g>
            <circle cx={byId[source].x} cy={byId[source].y} r={radii[source] + 9} fill="none" stroke={ink} strokeWidth={8} />
            <circle cx={byId[source].x} cy={byId[source].y} r={radii[source] + 9} fill="none" stroke={theme.colors.highlight ?? "#ffd54a"} strokeWidth={4} />
          </g>
        )}
      </svg>
      {tools &&
        createPortal(
          <button
            type="button"
            className={`play-switch${streaming ? " is-on" : ""}`}
            aria-pressed={streaming}
            onClick={() => setStreaming(!streaming)}
            title="When on, clicks start and stop streams, as Shift-clicks do"
          >
            Stream {streaming ? "on" : "off"}
          </button>,
          tools,
        )}
    </>
  );
}
