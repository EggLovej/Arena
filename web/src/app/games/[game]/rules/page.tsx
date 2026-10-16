import type { Metadata } from "next";

import { Markdown } from "@/components/Markdown.tsx";
import { getGame } from "@/lib/data.ts";

type Params = { params: Promise<{ game: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { game } = await params;
  return { title: `${getGame(game).name} rules` };
}

export default async function Rules({ params }: Params) {
  const { game: id } = await params;
  return <Markdown source={getGame(id).rules} />;
}
