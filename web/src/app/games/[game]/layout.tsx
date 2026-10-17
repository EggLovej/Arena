import type { ReactNode } from "react";

import { Explorer } from "@/components/Explorer.tsx";
import { listGames } from "@/lib/data.ts";

export const dynamicParams = false;

export function generateStaticParams() {
  return listGames().map((game) => ({ game: game.id }));
}

export default async function GameLayout({ children, params }: { children: ReactNode; params: Promise<{ game: string }> }) {
  const { game: id } = await params;
  return <Explorer gameId={id}>{children}</Explorer>;
}
