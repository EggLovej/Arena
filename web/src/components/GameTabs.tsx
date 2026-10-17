"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  ["", "Overview"],
  ["/watch", "Watch"],
  ["/leaderboard", "Leaderboard"],
  ["/bots", "Bots"],
  ["/maps", "Maps"],
  ["/rules", "Rules"],
] as const;

/** home: shown on the home page, which is the first game's overview. */
export function GameTabs({ gameId, home = false }: { gameId: string; home?: boolean }) {
  const pathname = usePathname();
  const base = `/games/${gameId}`;
  return (
    <nav className="game-tabs" aria-label="Game sections">
      {TABS.map(([path, label]) => {
        const href = base + path;
        const current = path === "" ? home || pathname === base : pathname === href || (path === "/watch" && pathname.startsWith(`${base}/matches/`));
        return (
          <Link key={path} href={href} aria-current={current ? "page" : undefined}>
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
