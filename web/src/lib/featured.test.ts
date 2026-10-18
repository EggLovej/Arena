import assert from "node:assert/strict";
import { test } from "node:test";

import { featuredMatches, liveMatch } from "./featured.ts";
import type { Leaderboard, MatchRow } from "./types.ts";

const order = ["ace", "bee", "cat", "dog"];
const leaderboard = {
  tables: { "2": { entries: order.map((bot) => ({ bot })) }, "3": { entries: order.map((bot) => ({ bot })) } },
} as unknown as Leaderboard;

function row(map: string, ...bots: string[]): MatchRow {
  const seats = ["red", "blue", "green"];
  const lineup = Object.fromEntries(bots.map((bot, index) => [seats[index], bot]));
  return { id: `${map}.${bots.join(".")}.0`, map, lineup, ranking: [["red"]], ticks: 100, replay: true };
}

test("strong lineups first, each map and each set of bots once, with the largest maps included", () => {
  const rows = [
    ...["m1", "m2", "m3", "m4"].flatMap((map) => [row(map, "ace", "bee"), row(map, "ace", "cat"), row(map, "bee", "dog")]),
    row("big1", "ace", "bee", "cat"),
    row("big2", "ace", "bee", "dog"),
    row("big3", "bee", "cat", "dog"),
  ];
  const chosen = featuredMatches(rows, leaderboard, 5);
  assert.deepEqual(
    chosen.map((match) => match.id),
    ["m1.ace.bee.0", "m2.ace.cat.0", "m3.bee.dog.0", "big1.ace.bee.cat.0", "big2.ace.bee.dog.0"],
  );
});

test("repeats a lineup only when there are not enough different ones", () => {
  const rows = ["m1", "m2", "m3"].map((map) => row(map, "ace", "bee"));
  assert.equal(featuredMatches(rows, leaderboard, 3).length, 3);
});

test("the live match is the strongest replay on the chosen map", () => {
  const rows = [row("m1", "ace", "bee"), row("m2", "cat", "dog"), row("m2", "ace", "cat"), row("m2", "bee", "dog")];
  assert.equal(liveMatch(rows, leaderboard, "m2")?.id, "m2.ace.cat.0");
  assert.equal(liveMatch(rows, leaderboard, "nowhere")?.id, "m1.ace.bee.0");
  assert.equal(liveMatch(rows, leaderboard)?.id, "m1.ace.bee.0");
  assert.equal(liveMatch(rows, null, "m2")?.id, "m2.cat.dog.0");
});
