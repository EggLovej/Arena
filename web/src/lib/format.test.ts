import assert from "node:assert/strict";
import { test } from "node:test";

import { describeOutcome, formatClock, formatDuration } from "./format.ts";
import type { Outcome } from "./types.ts";

const seats = {
  red: { id: "red", name: "Red", color: "#bb3933" },
  blue: { id: "blue", name: "Blue", color: "#2c5ba6" },
  green: { id: "green", name: "Green", color: "#36845b" },
};
const names = { strategic_ai: "Strategic AI", careful_ai: "Careful AI", aggressive_ai: "Aggressive AI" };
const lineup = { red: "strategic_ai", blue: "careful_ai", green: "aggressive_ai" };

test("times", () => {
  assert.equal(formatClock(125.3), "2:05.3");
  assert.equal(formatClock(59.96), "1:00.0"); // rounded before the minutes are counted
  assert.equal(formatDuration(48.4), "48 s");
  assert.equal(formatDuration(125), "2 min 05 s");
});

test("one sentence for how a match ended", () => {
  const cases: [Outcome, string][] = [
    [
      { ranking: [["red"], ["green"], ["blue"]], reason: "elimination", ticks: 1250 },
      "Strategic AI (Red) won by elimination after 2 min 05 s. Second: Aggressive AI (Green); Third: Careful AI (Blue).",
    ],
    [
      { ranking: [["red", "blue"], ["green"]], reason: "time", ticks: 1800 },
      "Draw between Strategic AI (Red) and Careful AI (Blue): time ran out after 3 min 00 s. Aggressive AI (Green) was knocked out.",
    ],
    // A game's own way of ending gets plain words.
    [{ ranking: [["blue"], ["red"]], reason: "finish", ticks: 90 }, "Careful AI (Blue) won after 9 s."],
    [{ ranking: [["red", "blue"]], reason: "finish", ticks: 90 }, "Draw between Strategic AI (Red) and Careful AI (Blue): they finished level after 9 s."],
  ];
  for (const [outcome, sentence] of cases) assert.equal(describeOutcome(outcome, lineup, names, seats, 0.1), sentence);
});
