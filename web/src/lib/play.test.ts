import assert from "node:assert/strict";
import { test } from "node:test";

import { PERSON, largerTurned, lineupFor, playsOn, result } from "./play.ts";

test("a seat wins alone, draws when it shares first place, and otherwise loses", () => {
  assert.equal(result({ ranking: [["red"], ["blue"]], reason: "elimination", ticks: 9 }, "red"), "won");
  assert.equal(result({ ranking: [["red"], ["blue"]], reason: "elimination", ticks: 9 }, "blue"), "lost");
  assert.equal(result({ ranking: [["red", "blue"], ["green"]], reason: "time", ticks: 9 }, "blue"), "drew");
  assert.equal(result({ ranking: [["red", "blue"], ["green"]], reason: "time", ticks: 9 }, "green"), "lost");
});

test("the person in their seat, and in every other the chosen bot if it plays the map, else one that does", () => {
  const atlas = { sits_out: ["Large maps"] };
  assert.equal(playsOn(atlas, { group: "Large maps" }), false);
  assert.equal(playsOn(atlas, { group: "" }), true);
  assert.equal(playsOn({}, { group: "Large maps" }), true);
  // seats, the person's seat (null: they watch), the chosen bots, the bots that play the map, and the lineup
  const cases: [string[], string | null, Record<string, string>, string[], Record<string, string>][] = [
    [["red", "blue", "green"], "blue", { red: "apex" }, ["apex", "strategic"], { red: "apex", blue: PERSON, green: "strategic" }],
    [["red", "blue"], "red", { blue: "atlas" }, ["strategic", "careful"], { red: PERSON, blue: "strategic" }], // the fallback plays instead
    [["red", "blue"], "red", { blue: "atlas" }, ["careful"], { red: PERSON, blue: "careful" }], // the fallback sits out too
    [["red", "blue"], null, { red: "apex" }, ["apex", "strategic"], { red: "apex", blue: "strategic" }],
  ];
  for (const [seats, person, bots, playable, lineup] of cases) assert.deepEqual(lineupFor(seats, person, bots, playable, "strategic"), lineup);
});

test("a wide board turns on a tall screen, and stays as it is on a wide one or when it is nearly square", () => {
  assert.equal(largerTurned([1000, 600], [380, 620]), true);
  assert.equal(largerTurned([1000, 600], [800, 330]), false);
  assert.equal(largerTurned([1000, 950], [380, 620]), false);
});
