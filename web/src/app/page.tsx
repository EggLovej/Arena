import { Explorer } from "@/components/Explorer.tsx";
import { listGames } from "@/lib/data.ts";

export default function Home() {
  if (!listGames().length) return <p className="notice page">No games have been published yet.</p>;
  return <Explorer />;
}
