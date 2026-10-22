// Small helpers for showing results and times the same way everywhere.

import type { Outcome, Seat } from "./types.ts";

/** Game time as m:ss.s, for example 2:05.3. */
export function formatClock(seconds: number): string {
  const tenths = Math.round(seconds * 10); // rounded first, so 59.96 s shows as 1:00.0
  const minutes = Math.floor(tenths / 600);
  return `${minutes}:${((tenths - minutes * 600) / 10).toFixed(1).padStart(4, "0")}`;
}

/** Game time as words: "2 min 05 s" or "48 s". */
export function formatDuration(seconds: number): string {
  const whole = Math.round(seconds);
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  return minutes ? `${minutes} min ${String(rest).padStart(2, "0")} s` : `${rest} s`;
}

export function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function listNames(names: string[]): string {
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

/** One sentence about how a match ended, for example "Strategic (Red) won by elimination after 2 min 05 s." */
export function describeOutcome(
  outcome: Outcome,
  lineup: Record<string, string>,
  botNames: Record<string, string>,
  seats: Record<string, Seat>,
  tickSeconds: number,
): string {
  const who = (seat: string) => `${botNames[lineup[seat]] ?? lineup[seat]} (${seats[seat]?.name ?? seat})`;
  const when = formatDuration(outcome.ticks * tickSeconds);
  const [first = [], ...rest] = outcome.ranking;
  // The places after the first, counted as in a race: two sharing first place make the next one third.
  let ahead = first.length;
  const places = rest.map((group) => {
    const place = ordinal(ahead + 1);
    ahead += group.length;
    return `${place}: ${listNames(group.map(who))}`;
  });
  if (first.length === 1) {
    const how = { elimination: " by elimination", time: " on time" }[outcome.reason] ?? "";
    const others = places.length > 1 ? ` ${places.join("; ")}.` : "";
    return `${who(first[0])} won${how} after ${when}.${others}`;
  }
  const others = places.length ? ` ${places.join("; ")}.` : "";
  const ending =
    outcome.reason === "time"
      ? `time ran out after ${when}`
      : outcome.reason === "elimination"
        ? `they fell at the same moment, after ${when}`
        : `they finished level after ${when}`;
  return `Draw between ${listNames(first.map(who))}: ${ending}.${others}`;
}

function ordinal(place: number): string {
  return ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"][place - 1] ?? `${place}th`;
}
