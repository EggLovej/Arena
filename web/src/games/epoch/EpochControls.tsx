"use client";

// Playing Epoch: a row of buttons over the sky, one for each action, with its price, and a
// key for each (1, 2 and 3 train units, T builds a turret, S sells one, A advances the age).
// Each press is one action; WebPerson in games/epoch/arena_game.py plays them in turn, one
// a tick. A button the side can't use now is faded, but pressing it still asks, so the
// page says why it was refused.

import { useEffect, useRef } from "react";

import { typing } from "@/components/usePlayback.ts";

import type { PlayControlsProps } from "../types.ts";
import { SIZES, SPRITES, type EpochBoardData, type EpochTheme, type SideState } from "./EpochBoard.tsx";

type Choice = { action: string; key: string; name: string; icon: string; size: [number, number]; cost: string; refusal: string | null };

const WIDTH = 62;
const GAP = 6;

/** Every action as the person's side stands: what it costs and, if the game would refuse it, why
 * (as price() and refusal() in games/epoch/epoch/battle.py work them out; the game has the last word). */
function choices(board: EpochBoardData, snapshot: PlayControlsProps["snapshot"], seat: string): Choice[] {
  const [age, , gold, xp, places] = (snapshot.sides?.[seat] ?? [0, 0, 0, 0, 1]) as SideState;
  const waiting = ((snapshot.queues?.[seat] as unknown[] | undefined)?.length ?? 1) - 1;
  const built = Object.entries((snapshot.turrets ?? {}) as Record<string, [string, number]>)
    .filter(([key]) => key.startsWith(`${seat}-`))
    .map(([key, [id]]) => ({ slot: Number(key.slice(seat.length + 1)), id }));
  const ageOf = (turret: string) => board.ages.findIndex((entry) => entry.turret === turret);
  const afford = (price: number) => (gold < price ? `that costs ${price} gold` : null);

  const units = board.ages[age].units.map((id, index): Choice => {
    const { name, role, price } = board.units[id];
    const refusal = waiting >= board.queue ? `${board.queue} units are already waiting` : afford(price);
    return { action: role, key: String(index + 1), name, icon: `${id}-walk-1`, size: SIZES[role], cost: String(price), refusal };
  });
  const turret = board.ages[age].turret;
  const opening = built.length === places && places < board.places.length;
  const turretPrice = board.turrets[turret].price + (opening ? board.places[places] : 0);
  // Selling sells the oldest kind of turret, and of those the last built (_to_sell in epoch/battle.py).
  const sold = built.sort((a, b) => ageOf(a.id) - ageOf(b.id) || b.slot - a.slot)[0];
  const coming = board.ages[age + 1];
  return [
    ...units,
    {
      action: "turret",
      key: "T",
      name: opening ? `${board.turrets[turret].name} (opens a new place)` : board.turrets[turret].name,
      icon: turret,
      size: [40, 40],
      cost: String(turretPrice),
      refusal: built.length === board.places.length ? "every turret place is taken" : afford(turretPrice),
    },
    {
      action: "sell",
      key: "S",
      name: sold ? `Sell a ${board.turrets[sold.id].name}` : "Sell a turret",
      icon: sold?.id ?? turret,
      size: [40, 40],
      cost: sold ? `+${Math.floor(board.turrets[sold.id].price / 2)}` : "",
      refusal: sold ? null : "there is no turret to sell",
    },
    {
      action: "age",
      key: "A",
      name: coming ? `Advance to the ${coming.name}` : "The last age",
      icon: `base_${(coming ?? board.ages[age]).id}`,
      size: [120, 150],
      cost: coming ? String(coming.price) : "",
      refusal: !coming ? "this is the last age" : xp < coming.xp ? `the ${coming.name} needs ${coming.xp} XP` : afford(coming.price),
    },
  ];
}

export function EpochControls({ board: rawBoard, snapshot, theme: rawTheme, seat, active, send, layer }: PlayControlsProps) {
  const board = rawBoard as EpochBoardData;
  const theme = rawTheme as EpochTheme;
  const color = theme.players[seat] ?? theme.ink;
  const options = choices(board, snapshot, seat);
  const left = (board.length - options.length * WIDTH - (options.length - 1) * GAP) / 2;

  const latest = useRef({ active, send });
  useEffect(() => {
    latest.current = { active, send };
  });
  const press = (action: string) => {
    if (latest.current.active) latest.current.send(action);
  };

  useEffect(() => {
    const keys: Record<string, string> = { "1": "melee", "2": "ranged", "3": "heavy", t: "turret", s: "sell", a: "age" };
    const down = (event: KeyboardEvent) => {
      if (typing(event) || event.repeat) return;
      const action = keys[event.key.toLowerCase()];
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
        const scale = 34 / Math.max(w, h);
        return (
          <g
            key={action}
            transform={`translate(${left + index * (WIDTH + GAP)} 8)`}
            role="button"
            aria-label={`${name}${cost ? `, ${cost} gold` : ""} (${key})`}
            aria-disabled={refusal !== null}
            opacity={refusal ? 0.5 : 1}
            style={{ cursor: active ? "pointer" : "default" }}
            onClick={() => press(action)}
          >
            <title>{refusal ? `${name}: ${refusal}` : `${name} (${key})`}</title>
            <rect width={WIDTH} height={60} rx={8} fill={theme.panel} stroke={theme.ink} strokeWidth={1.5} />
            <use href={`${SPRITES}#${icon}`} x={(WIDTH - w * scale) / 2} y={6} width={w * scale} height={h * scale} style={{ color }} />
            <text x={6} y={14} fontSize={11} fontWeight={700} fill={theme.ink} opacity={0.6}>
              {key}
            </text>
            <text x={WIDTH / 2} y={54} fontSize={13} fontWeight={700} fill={theme.ink} textAnchor="middle">
              {cost}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
