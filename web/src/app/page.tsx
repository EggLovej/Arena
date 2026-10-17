import { Explorer } from "@/components/Explorer.tsx";
import { GameOverview } from "@/components/GameOverview.tsx";
import { listGames } from "@/lib/data.ts";

// The home page is the first game's overview, with every game in the list beside it.
export default function Home() {
  const [first] = listGames();
  if (!first) return <p className="notice page">No games have been published yet.</p>;
  return (
    <Explorer gameId={first.id} home>
      <GameOverview gameId={first.id} />
    </Explorer>
  );
}
