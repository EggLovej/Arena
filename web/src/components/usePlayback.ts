"use client";

// What every replay on the site shares: loading one, and a clock that moves it
// on with the screen's refresh.

import { useEffect, useMemo, useRef, useState } from "react";

import { decodeFrames } from "@/lib/frames.ts";
import type { Replay, Snapshot } from "@/lib/types.ts";

export function useReplay(url: string): { replay: Replay | null; snapshots: Snapshot[]; error: string | null } {
  const [replay, setReplay] = useState<Replay | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(url)
      .then((response) => {
        if (!response.ok) throw new Error(`The replay could not be loaded (${response.status}).`);
        return response.json() as Promise<Replay>;
      })
      .then((data) => !cancelled && setReplay(data))
      .catch((reason: Error) => !cancelled && setError(reason.message));
    return () => {
      cancelled = true;
    };
  }, [url]);

  const snapshots = useMemo<Snapshot[]>(() => (replay ? decodeFrames(replay.frames) : []), [replay]);
  return { replay, snapshots, error };
}

/** While running, calls advance with the seconds since the last frame, so fleets glide between ticks. */
export function useAnimationClock(running: boolean, advance: (seconds: number) => void) {
  const latest = useRef(advance);
  useEffect(() => {
    latest.current = advance;
  });
  useEffect(() => {
    if (!running) return;
    // A frame's timestamp can be a little before performance.now(), so time starts at the first frame.
    let previous: number | null = null;
    let frame = 0;
    const step = (now: number) => {
      latest.current(previous === null ? 0 : Math.max(0, (now - previous) / 1000));
      previous = now;
      frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [running]);
}
