// Rebuilding every tick's snapshot from a replay's keyframes and deltas.
// The same format as platform/arena/frames.py: a delta lists, per group, the
// entries that changed (their new value) or were removed (null).

import type { Delta, Frames, Snapshot } from "./types.ts";

export function applyDelta(snapshot: Snapshot, delta: Delta): Snapshot {
  const result: Snapshot = { ...snapshot };
  for (const [group, entries] of Object.entries(delta)) {
    const target = { ...(result[group] ?? {}) };
    for (const [key, value] of Object.entries(entries)) {
      if (value === null) delete target[key];
      else target[key] = value;
    }
    result[group] = target;
  }
  return result;
}

/** Every tick's snapshot, in order. Groups that did not change are shared between ticks. */
export function decodeFrames(frames: Frames): Snapshot[] {
  const snapshots: Snapshot[] = [];
  let current: Snapshot | undefined;
  for (let tick = 0; tick < frames.count; tick++) {
    const keyframe = frames.keyframes[String(tick)];
    if (keyframe) current = keyframe;
    else if (current) current = applyDelta(current, frames.deltas[tick - 1]);
    else throw new Error(`Replay has no keyframe before tick ${tick}`);
    snapshots.push(current);
  }
  return snapshots;
}
