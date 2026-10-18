"use client";

// Playing Graphite with the mouse or a finger, as in the game's own window
// (games/graphite/human_controller.py): click one of your bases, then a base next to
// it, and half its units go there. Shift-click the second base instead, or turn on
// Stream, and from then on everything that base has goes there, tick after tick, until
// you shift-click it again. With a keyboard, Tab moves between the bases and Enter (or
// Shift+Enter) does what a click (or Shift-click) would. The commands go to WebPerson in
// games/graphite/arena_game.py.

import { useEffect, useRef, useState, type KeyboardEvent as KeyPress, type MouseEvent } from "react";
import { createPortal } from "react-dom";

import { typing } from "@/components/usePlayback.ts";

import type { PlayControlsProps } from "../types.ts";
import { LARGE_MAP, baseRadii, formatRate, type BaseState, type GraphiteBoardData } from "./GraphiteBoard.tsx";

type Theme = { colors: Record<string, string>; players: Record<string, { fill: string; dark: string }> };

// How far outside a base's circle a click still counts, in map units: a finger is less precise than a mouse.
const MARGIN = { mouse: 6, touch: 16 };

export function GraphiteControls({ board: rawBoard, snapshot, theme: rawTheme, seat, person, active, send, tools, layer, toBoard }: PlayControlsProps) {
  const board = rawBoard as GraphiteBoardData;
  const theme = rawTheme as Theme;
  const [selected, setSelected] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false); // the Stream switch, for screens without a Shift key
  const touch = useRef(false);

  const bases = (snapshot.bases ?? {}) as Record<string, BaseState>;
  const streams = ((person as { streams?: Record<string, string> } | undefined)?.streams ?? {}) as Record<string, string>;
  const radii = baseRadii(board);
  // Large maps pack small bases close together, so the rings around them sit closer too.
  const large = board.bases.length >= LARGE_MAP;
  const ring = large ? { target: 5, selected: 6, key: 3, stream: 4 } : { target: 7, selected: 9, key: 13, stream: 6 };
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
      if (event.key === "Escape" && !typing(event)) setSelected(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const baseAt = (event: MouseEvent): string | null => {
    const point = toBoard(event);
    if (!point) return null;
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

  // What a click on a base (or beside every base, for null) does: Graphite's own window's rules.
  const choose = (clicked: string | null, stream: boolean) => {
    if (!active) return;
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

  const click = (event: MouseEvent) => choose(baseAt(event), event.shiftKey || streaming);

  const press = (id: string) => (event: KeyPress) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    choose(id, event.shiftKey || streaming);
  };

  const describe = (id: string) => {
    const [owner, units] = bases[id] ?? ["", 0];
    const whose = owner === seat ? "yours" : owner ? `${owner}'s` : "nobody's";
    const role = id === source ? ", selected" : source && neighbours[source]?.has(id) ? ", next to the selected base" : "";
    return `${id.replaceAll("_", " ")}: ${whose}, ${units} units${streams[id] ? `, streaming to ${streams[id].replaceAll("_", " ")}` : ""}${role}`;
  };

  const pointer = hovered && (source ? neighbours[source]?.has(hovered) || mine(hovered) : mine(hovered) || streams[hovered]);

  return (
    <>
      <svg
        {...layer}
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
        aria-label="The board: pick one of your bases, then a base next to it, to send half its units"
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
              strokeWidth={ring.stream}
              markerEnd="url(#stream-head)"
            />
          );
        })}
        {source &&
          [...(neighbours[source] ?? [])].map((id) => (
            <circle key={`target-${id}`} className="play-target" cx={byId[id].x} cy={byId[id].y} r={radii[id] + ring.target} stroke={colour} />
          ))}
        {hovered && active && hovered !== source && (source ? neighbours[source]?.has(hovered) : mine(hovered)) && (
          <circle cx={byId[hovered].x} cy={byId[hovered].y} r={radii[hovered] + ring.target} fill="none" stroke={ink} strokeWidth={3} opacity={0.5} />
        )}
        {source && (
          <g>
            <circle cx={byId[source].x} cy={byId[source].y} r={radii[source] + ring.selected} fill="none" stroke={ink} strokeWidth={large ? 6 : 8} />
            <circle cx={byId[source].x} cy={byId[source].y} r={radii[source] + ring.selected} fill="none" stroke={theme.colors.highlight ?? "#ffd54a"} strokeWidth={large ? 3 : 4} />
          </g>
        )}
        {board.bases.map((base) => (
          <circle
            key={`key-${base.id}`}
            className="play-base-key"
            cx={base.x}
            cy={base.y}
            r={radii[base.id] + ring.key}
            stroke={ink}
            tabIndex={active ? 0 : -1}
            role="button"
            aria-label={describe(base.id)}
            onKeyDown={press(base.id)}
            onMouseDown={(event) => event.preventDefault()} // a click does not take the keyboard's focus, so Space still pauses
          >
            {/* Large maps show only units on the bases; the rest shows on hover, as on the board. */}
            {large && <title>{`${base.id} · ${formatRate(base.production)}`}</title>}
          </circle>
        ))}
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
