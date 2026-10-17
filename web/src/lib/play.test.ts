import assert from "node:assert/strict";
import { test } from "node:test";

import { PERSON, drawnTick, lineupFor, result } from "./play.ts";

test("the board is drawn between ticks, at most one tick ahead", () => {
  assert.equal(drawnTick(10, 1000, 1050, 100, true), 10.5);
  assert.equal(drawnTick(10, 1000, 1400, 100, true), 11);
  assert.equal(drawnTick(10, 1000, 990, 100, true), 10);
  assert.equal(drawnTick(10, 1000, 1050, 100, false), 10);
});

test("a seat wins alone, draws when it shares first place, and otherwise loses", () => {
  assert.equal(result({ ranking: [["red"], ["blue"]], reason: "elimination", ticks: 9 }, "red"), "won");
  assert.equal(result({ ranking: [["red"], ["blue"]], reason: "elimination", ticks: 9 }, "blue"), "lost");
  assert.equal(result({ ranking: [["red", "blue"], ["green"]], reason: "time", ticks: 9 }, "blue"), "drew");
  assert.equal(result({ ranking: [["red", "blue"], ["green"]], reason: "time", ticks: 9 }, "green"), "lost");
});

test("the lineup puts the person in their seat and a bot everywhere else", () => {
  assert.deepEqual(lineupFor(["red", "blue", "green"], "blue", { red: "apex" }, "strategic"), {
    red: "apex",
    blue: PERSON,
    green: "strategic",
  });
});
