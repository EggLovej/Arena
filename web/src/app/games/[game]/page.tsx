import type { Metadata } from "next";

import { GameOverview } from "@/components/GameOverview.tsx";
import { getGame } from "@/lib/data.ts";

type Params = { params: Promise<{ game: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { game } = await params;
  return { title: getGame(game).name };
}

export default async function GameOverviewPage({ params }: Params) {
  const { game: id } = await params;
  return <GameOverview gameId={id} />;
}
