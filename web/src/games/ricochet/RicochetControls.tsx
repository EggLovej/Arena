"use client";

// Playing Ricochet with a keyboard and mouse, as in the game's own window
// (games/ricochet/src/ricochet/display/human.py): W A S D move, the tank aims at the
// pointer, the left button fires and the right button or Space raises the shield. The
// page sends what is held whenever it changes, and WebPerson in
// games/ricochet/arena_game.py holds it until the next change.

import { useEffect, useRef, useState, type PointerEvent } from "react";

import { typing } from "@/components/usePlayback.ts";

import type { PlayControlsProps } from "../types.ts";
import type { RicochetBoardData } from "./RicochetBoard.tsx";

type Theme = { players: Record<string, string>; text: string };
type Held = { move: [number, number]; target: [number, number] | null; fire: boolean; shield: boolean };

const MOVES: Record<string, [number, number]> = {
  KeyW: [0, -1],
  KeyA: [-1, 0],
  KeyS: [0, 1],
  KeyD: [1, 0],
  ArrowUp: [0, -1],
  ArrowLeft: [-1, 0],
  ArrowDown: [0, 1],
  ArrowRight: [1, 0],
};

export function RicochetControls({ board: rawBoard, theme: rawTheme, seat, active, send }: PlayControlsProps) {
  const board = rawBoard as RicochetBoardData;
  const theme = rawTheme as Theme;
  const svg = useRef<SVGSVGElement>(null);
  const [target, setTarget] = useState<[number, number] | null>(null);
  const colour = theme.players[seat] ?? theme.text;

  // What is held, kept outside React's state: it changes with every key and pointer move.
  const keys = useRef(new Set<string>());
  const held = useRef({ target: null as [number, number] | null, fire: false, shield: false });
  const latest = useRef({ active, send });
  useEffect(() => {
    latest.current = { active, send };
  });

  // Sent at most once a frame, and only when something changed since the last time.
  const sent = useRef("");
  const frame = useRef(0);
  const flush = useRef(() => {
    frame.current = 0;
    if (!latest.current.active) return;
    let [x, y] = [0, 0];
    for (const code of keys.current) {
      x += MOVES[code]?.[0] ?? 0;
      y += MOVES[code]?.[1] ?? 0;
    }
    const length = Math.hypot(x, y);
    const command: Held = {
      move: length > 1 ? [x / length, y / length] : [x, y],
      target: held.current.target,
      fire: held.current.fire,
      shield: held.current.shield || keys.current.has("Space"),
    };
    const text = JSON.stringify(command);
    if (text !== sent.current) {
      sent.current = text;
      latest.current.send(command);
    }
  });
  const changed = () => {
    if (!frame.current) frame.current = requestAnimationFrame(flush.current);
  };

  // Coming back from a pause or a countdown, the match learns what is held now.
  useEffect(() => {
    if (!active) return;
    sent.current = "";
    changed();
  }, [active]);

  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      if (typing(event)) return;
      if (!(event.code in MOVES) && event.code !== "Space") return;
      // Keys are followed all along, so a key held through the countdown counts from the start. While the
      // clock runs they are the game's: the arrows and Space do not scroll the page, and Space (the
      // shield) does not press a button that has the focus.
      if (latest.current.active) {
        event.preventDefault();
        if (event.code === "Space" && document.activeElement instanceof HTMLElement) document.activeElement.blur();
      }
      keys.current.add(event.code);
      changed();
    };
    const up = (event: KeyboardEvent) => {
      if (keys.current.delete(event.code)) changed();
    };
    // When the window loses the keyboard it hears no more key-ups, so everything is let go.
    const away = () => {
      keys.current.clear();
      held.current = { ...held.current, fire: false, shield: false };
      changed();
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", away);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", away);
      cancelAnimationFrame(frame.current);
      frame.current = 0;
    };
  }, []);

  // Every pointer event says where it is and which buttons are down; a finger fires while it touches.
  const point = (event: PointerEvent) => {
    const matrix = svg.current?.getScreenCTM();
    if (!matrix) return;
    const at = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    const aimed: [number, number] = [Math.round(Math.min(board.width, Math.max(0, at.x))), Math.round(Math.min(board.height, Math.max(0, at.y)))];
    const touching = event.type !== "pointerup" && event.type !== "pointercancel";
    held.current = {
      target: aimed,
      fire: event.pointerType === "mouse" ? (event.buttons & 1) !== 0 : touching,
      shield: event.pointerType === "mouse" && (event.buttons & 2) !== 0,
    };
    setTarget(aimed);
    changed();
  };

  return (
    <svg
      ref={svg}
      viewBox={`0 0 ${board.width} ${board.height}`}
      className="play-controls-layer"
      style={{ cursor: active ? "none" : "default", touchAction: "none" }}
      onPointerMove={point}
      onPointerDown={(event) => {
        // The focus leaves any button, so Space raises the shield rather than pressing the button.
        if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
        event.currentTarget.setPointerCapture(event.pointerId);
        point(event);
      }}
      onPointerUp={point}
      onPointerCancel={point}
      onContextMenu={(event) => event.preventDefault()}
      role="application"
      aria-label="The arena: W A S D move your tank, it aims at the pointer, the left button fires and the right button or Space raises the shield"
    >
      {active && target && (
        <g transform={`translate(${target[0]} ${target[1]})`} stroke={colour} strokeWidth={2} fill="none" pointerEvents="none">
          <circle r={11} />
          <path d="M -18 0 H -6 M 6 0 H 18 M 0 -18 V -6 M 0 6 V 18" strokeLinecap="round" />
        </g>
      )}
    </svg>
  );
}
