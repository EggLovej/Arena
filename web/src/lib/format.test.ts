import assert from "node:assert/strict";
import { test } from "node:test";

import { describeOutcome, formatClock, formatDuration, listNames } from "./format.ts";

const seats = {
  red: { id: "red", name: "Red", color: "#bb3933" },
  blue: { id: "blue", name: "Blue", color: "#2c5ba6" },
  green: { id: "green", name: "Green", color: "#36845b" },
};
const names = { strategic_ai: "Strategic AI", careful_ai: "Careful AI", aggressive_ai: "Aggressive AI" };
const lineup = { red: "strategic_ai", blue: "careful_ai", green: "aggressive_ai" };

test("times", () => {
  assert.equal(formatClock(125.3), "2:05.3");
  assert.equal(formatClock(4), "0:04.0");
  assert.equal(formatClock(59.96), "1:00.0");
  assert.equal(formatClock(119.97), "2:00.0");
  assert.equal(formatClock(59.94), "0:59.9");
  assert.equal(formatDuration(48.4), "48 s");
  assert.equal(formatDuration(125), "2 min 05 s");
  assert.equal(listNames(["a", "b", "c"]), "a, b and c");
});

test("a win, with the other places", () => {
  const text = describeOutcome({ ranking: [["red"], ["green"], ["blue"]], reason: "elimination", ticks: 1250 }, lineup, names, seats, 0.1);
  assert.equal(text, "Strategic AI (Red) won by elimination after 2 min 05 s. Second: Aggressive AI (Green); Third: Careful AI (Blue).");
});

test("a draw on time", () => {
  const text = describeOutcome({ ranking: [["red", "blue"], ["green"]], reason: "time", ticks: 1800 }, lineup, names, seats, 0.1);
  assert.equal(text, "Draw between Strategic AI (Red) and Careful AI (Blue): time ran out after 3 min 00 s. Aggressive AI (Green) was knocked out.");
});

test("a game's own way of ending gets plain words", () => {
  const won = describeOutcome({ ranking: [["blue"], ["red"]], reason: "finish", ticks: 90 }, lineup, names, seats, 0.1);
  assert.equal(won, "Careful AI (Blue) won after 9 s.");
  const level = describeOutcome({ ranking: [["red", "blue"]], reason: "finish", ticks: 90 }, lineup, names, seats, 0.1);
  assert.equal(level, "Draw between Strategic AI (Red) and Careful AI (Blue): they finished level after 9 s.");
});
