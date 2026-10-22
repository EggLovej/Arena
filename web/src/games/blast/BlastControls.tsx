"use client";

// Playing Blast with a keyboard, a mouse or a finger. The arrow keys or W A S D walk: of
// two held, the one pressed last wherever it is open and the other where it isn't, which
// takes corners. Space or the Bomb button drops a bomb. Pressing on the board walks
// towards the pointer while it is down, and pressing on your own bomber drops a bomb. The
// page sends the ways to walk whenever they change, and each bomb; WebPerson in
// games/blast/arena_game.py keeps walking them until told otherwise.

import { useEffect, useRef, type PointerEvent } from "react";
import { createPortal } from "react-dom";

import { typing } from "@/components/usePlayback.ts";

import type { PlayControlsProps } from "../types.ts";
import { TILE, type BomberState } from "./BlastBoard.tsx";

type Way = "up" | "down" | "left" | "right";

const KEYS: Record<string, Way> = {
  ArrowUp: "up",
  ArrowDown: "down",
  ArrowLeft: "left",
  ArrowRight: "right",
  KeyW: "up",
  KeyS: "down",
  KeyA: "left",
  KeyD: "right",
};

export function BlastControls({ snapshot, seat, active, send, tools, layer, toBoard }: PlayControlsProps) {
  const me = ((snapshot.bombers ?? {}) as Record<string, BomberState>)[seat];

  // What is held, kept outside React's state: it changes with every key and pointer move.
  const keys = useRef<Way[]>([]); // the walking keys held, the last pressed last
  const pointer = useRef<DOMPoint | null>(null); // where the pointer is held down on the board
  const latest = useRef({ active, send, me });
  useEffect(() => {
    latest.current = { active, send, me };
  });

  // The ways to walk are sent only when they change.
  const sent = useRef<string | null>(null);
  const update = useRef(() => {
    const { active, send, me } = latest.current;
    if (!active) return;
    let ways = keys.current.toReversed();
    if (pointer.current && me) {
      // Towards the pointer, first along whichever axis it is further away on, until the bomber is under it.
      const dx = pointer.current.x - (me[0] + 0.5) * TILE;
      const dy = pointer.current.y - (me[1] + 0.5) * TILE;
      const across: Way[] = Math.abs(dx) > TILE / 2 ? [dx > 0 ? "right" : "left"] : [];
      const along: Way[] = Math.abs(dy) > TILE / 2 ? [dy > 0 ? "down" : "up"] : [];
      ways = Math.abs(dx) > Math.abs(dy) ? [...across, ...along] : [...along, ...across];
    }
    const text = ways.join();
    if (text !== sent.current) {
      sent.current = text;
      send({ walk: ways });
    }
  });
  const bomb = () => {
    if (latest.current.active) latest.current.send("bomb");
  };

  // Coming back from a pause or a countdown, the match learns what is held now; and a held
  // pointer is walked towards from wherever the bomber has got to.
  useEffect(() => {
    if (!active) return;
    sent.current = null;
    update.current();
  }, [active]);
  useEffect(() => {
    if (pointer.current) update.current();
  }, [me]);

  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      if (typing(event)) return;
      const way = KEYS[event.code];
      if (!way && event.code !== "Space") return;
      // While the clock runs the keys are the game's: the arrows and Space do not scroll the
      // page, and Space does not press a button that has the focus.
      if (latest.current.active) event.preventDefault();
      if (event.repeat) return;
      if (way) {
        keys.current = [...keys.current.filter((held) => held !== way), way];
        update.current();
      } else bomb();
    };
    const up = (event: KeyboardEvent) => {
      const way = KEYS[event.code];
      if (!way) return;
      keys.current = keys.current.filter((held) => held !== way);
      update.current();
    };
    // When the window loses the keyboard it hears no more key-ups, so everything is let go.
    const away = () => {
      keys.current = [];
      pointer.current = null;
      update.current();
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", away);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", away);
    };
  }, []);

  const point = (event: PointerEvent) => {
    pointer.current = event.type === "pointerup" || event.type === "pointercancel" ? null : toBoard(event);
    update.current();
  };

  return (
    <>
      <svg
        {...layer}
        style={{ touchAction: "none" }}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          const at = toBoard(event);
          if (at && me && Math.abs(at.x - (me[0] + 0.5) * TILE) < TILE / 2 && Math.abs(at.y - (me[1] + 0.5) * TILE) < TILE / 2) bomb();
          else point(event);
        }}
        onPointerMove={(event) => pointer.current && point(event)}
        onPointerUp={point}
        onPointerCancel={point}
        onContextMenu={(event) => event.preventDefault()}
        role="application"
        aria-label="The grid: the arrow keys or W A S D walk and Space drops a bomb; press on the board to walk towards the pointer, or on your bomber to drop a bomb"
      />
      {tools &&
        createPortal(
          <button type="button" onClick={bomb} onMouseDown={(event) => event.preventDefault()} disabled={!active}>
            Bomb
          </button>,
          tools,
        )}
    </>
  );
}
