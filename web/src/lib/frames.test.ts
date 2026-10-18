// frames.ts must read replays exactly as platform/arena/frames.py writes them.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { applyDelta, decodeFrames } from "./frames.ts";
import type { Frames, Snapshot } from "./types.ts";

const fixture = JSON.parse(readFileSync(new URL("./fixtures/frames.json", import.meta.url), "utf8")) as {
  frames: Frames;
  snapshots: Snapshot[];
};

test("decodes every tick the Python recorder stored", () => {
  const decoded = decodeFrames(fixture.frames);
  assert.equal(decoded.length, fixture.snapshots.length);
  decoded.forEach((snapshot, tick) => assert.deepEqual(snapshot, fixture.snapshots[tick], `tick ${tick}`));
});

test("a null entry removes it and leaves the earlier snapshot alone", () => {
  const before: Snapshot = { fleets: { "1": [1], "2": [2] } };
  const after = applyDelta(before, { fleets: { "1": null, "3": [3] } });
  assert.deepEqual(after, { fleets: { "2": [2], "3": [3] } });
  assert.deepEqual(before, { fleets: { "1": [1], "2": [2] } });
});

test("a replay without a first keyframe is refused", () => {
  assert.throws(() => decodeFrames({ count: 2, keyframes: {}, deltas: [{}] }));
});
