"use client";

// Playing Epoch: a row of buttons over the sky, one for each action, with its price, and a
// key for each: 1 to 4 train units, Q, W and E build the age's three turrets, R opens a turret
// place, S sells a turret, A advances the age and X calls the special. Each press is one
// action; WebPerson in games/epoch/arena_game.py plays them in turn, one a tick. A button the
// side can't use now is faded, but pressing it still asks, so the page says why it was refused.

import { useEffect, useRef } from "react";

import { typing } from "@/components/usePlayback.ts";

import type { PlayControlsProps } from "../types.ts";
import { SIZES, SPRITES, type EpochBoardData, type EpochTheme, type SideState } from "./EpochBoard.tsx";

type Choice = {
  action: string;
  key: string;
  name: string;
  icon: string | null; // a sprite, or none for an action drawn here
  size: [number, number];
  cost: string;
  refusal: string | null;
};

const WIDTH = 56;
const GAP = 4;
const KEYS: Record<string, string> = { q: "turret1", w: "turret2", e: "turret3", r: "place", s: "sell", a: "age", x: "special" };
const ROLE_KEYS = ["melee", "ranged", "heavy", "super"]; // the keys 1 to 4

/** Every action as the person's side stands: what it costs and, if the game would refuse it, why
 * (as price() and refusal() in games/epoch/epoch/battle.py work them out; the game has the last word). */
function choices(board: EpochBoardData, snapshot: PlayControlsProps["snapshot"], seat: string, tick: number): Choice[] {
  const [age, , gold, xp, places, special] = (snapshot.sides?.[seat] ?? [0, 0, 0, 0, 1, 0]) as SideState;
  const waiting = ((snapshot.queues?.[seat] as unknown[] | undefined)?.length ?? 1) - 1;
  const built = Object.entries((snapshot.turrets ?? {}) as Record<string, [string, number]>)
    .filter(([key]) => key.startsWith(`${seat}-`))
    .map(([key, [id]]) => ({ slot: Number(key.slice(seat.length + 1)), id }));
  const afford = (price: number) => (gold < price ? `that costs ${price} gold` : null);
  const current = board.ages[age];

  const units = current.units.map((id): Choice => {
    const { name, role, price } = board.units[id];
    const refusal = waiting >= board.queue ? `${board.queue} units are already waiting` : afford(price);
    return { action: role, key: String(ROLE_KEYS.indexOf(role) + 1), name, icon: `${id}-walk-1`, size: SIZES[role], cost: String(price), refusal };
  });
  const free = built.length < places;
  const turrets = current.turrets.map((id, index): Choice => {
    const { name, price } = board.turrets[id];
    const refusal = free ? afford(price) : "no turret place is free";
    return { action: `turret${index + 1}`, key: "QWE"[index], name, icon: id, size: [40, 40], cost: String(price), refusal };
  });
  // Selling sells the cheapest turret, and of those the last built (_to_sell in epoch/battle.py).
  const sold = built.sort((a, b) => board.turrets[a.id].price - board.turrets[b.id].price || b.slot - a.slot)[0];
  const opening = board.places[places];
  const coming = board.ages[age + 1];
  const wait = special + board.special_cooldown - (tick + 1); // ticks until a special called now could strike
  return [
    ...units,
    ...turrets,
    {
      action: "place",
      key: "R",
      name: "Open a turret place",
      icon: null,
      size: [40, 40],
      cost: opening === undefined ? "" : String(opening),
      refusal: opening === undefined ? "every turret place is open" : afford(opening),
    },
    {
      action: "sell",
      key: "S",
      name: sold ? `Sell a ${board.turrets[sold.id].name}` : "Sell a turret",
      icon: sold?.id ?? current.turrets[0],
      size: [40, 40],
      cost: sold ? `+${Math.floor(board.turrets[sold.id].price / 2)}` : "",
      refusal: sold ? null : "there is no turret to sell",
    },
    {
      action: "age",
      key: "A",
      name: coming ? `Advance to the ${coming.name}` : "The last age",
      icon: `base_${(coming ?? current).id}`,
      size: [120, 150],
      cost: coming ? `${coming.xp} XP` : "",
      refusal: !coming ? "this is the last age" : xp < coming.xp ? `the ${coming.name} needs ${coming.xp} XP` : null,
    },
    {
      action: "special",
      key: "X",
      name: current.special.name,
      icon: null,
      size: [40, 40],
      cost: wait > 0 ? "" : "ready",
      refusal: wait > 0 ? `the ${current.special.name} is recharging` : null,
    },
  ];
}

export function EpochControls({ board: rawBoard, snapshot, tick, theme: rawTheme, seat, active, send, layer }: PlayControlsProps) {
  const board = rawBoard as EpochBoardData;
  const theme = rawTheme as EpochTheme;
  const color = theme.players[seat] ?? theme.ink;
  const options = choices(board, snapshot, seat, tick);
  const left = (board.length - options.length * WIDTH - (options.length - 1) * GAP) / 2;
  const special = ((snapshot.sides?.[seat] ?? []) as SideState)[5] ?? 0;
  const charge = Math.min(1, Math.max(0, (tick + 1 - special) / board.special_cooldown));

  const latest = useRef({ active, send });
  useEffect(() => {
    latest.current = { active, send };
  });
  const press = (action: string) => {
    if (latest.current.active) latest.current.send(action);
  };

  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      if (typing(event) || event.repeat) return;
      const key = event.key.toLowerCase();
      const action = KEYS[key] ?? ROLE_KEYS[Number(key) - 1];
      if (!action || !latest.current.active) return;
      event.preventDefault();
      latest.current.send(action);
    };
    window.addEventListener("keydown", down);
    return () => window.removeEventListener("keydown", down);
  }, []);

  return (
    <svg {...layer} preserveAspectRatio="xMidYMax meet" aria-label="Your actions">
      {options.map(({ action, key, name, icon, size: [w, h], cost, refusal }, index) => {
        const scale = 32 / Math.max(w, h);
        return (
          <g
            key={action}
            transform={`translate(${left + index * (WIDTH + GAP)} 8)`}
            role="button"
            aria-label={`${name}${cost ? `, ${cost}` : ""} (${key})`}
            aria-disabled={refusal !== null}
            opacity={refusal ? 0.5 : 1}
            style={{ cursor: active ? "pointer" : "default" }}
            onClick={() => press(action)}
          >
            <title>{refusal ? `${name}: ${refusal}` : `${name} (${key})`}</title>
            <rect width={WIDTH} height={60} rx={8} fill={theme.panel} stroke={theme.ink} strokeWidth={1.5} />
            {icon && <use href={`${SPRITES}#${icon}`} x={(WIDTH - w * scale) / 2} y={6} width={w * scale} height={h * scale} style={{ color }} />}
            {action === "place" && (
              <path d={`M${WIDTH / 2 - 10} 22h20M${WIDTH / 2} 12v20`} stroke={theme.ink} strokeWidth={4} strokeLinecap="round" />
            )}
            {action === "special" && (
              // A clock face that fills as the special recharges.
              <g transform={`translate(${WIDTH / 2} 22)`}>
                <circle r={13} fill={theme.panel} stroke={theme.ink} strokeWidth={1.5} />
                <path
                  d={charge >= 1 ? "M0 -13a13 13 0 1 1 0 26a13 13 0 1 1 0 -26z" : `M0 0L0 -13A13 13 0 ${charge > 0.5 ? 1 : 0} 1 ${13 * Math.sin(2 * Math.PI * charge)} ${-13 * Math.cos(2 * Math.PI * charge)}z`}
                  fill={color}
                />
              </g>
            )}
            <text x={5} y={13} fontSize={11} fontWeight={700} fill={theme.ink} opacity={0.6}>
              {key}
            </text>
            <text x={WIDTH / 2} y={54} fontSize={cost.length > 6 ? 10 : 12} fontWeight={700} fill={theme.ink} textAnchor="middle">
              {cost}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
