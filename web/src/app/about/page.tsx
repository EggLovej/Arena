import type { Metadata } from "next";

export const metadata: Metadata = { title: "About" };

export default function About() {
  return (
    <div className="prose">
      <h1>How Arena works</h1>
      <p className="lede">
        Arena hosts games that bots play. The bots are classic AI: hand-written strategies, search, and neural networks
        trained by playing themselves. No language models.
      </p>

      <h2>Matches</h2>
      <p>
        Every game runs in fixed ticks, and every bot sees the same state and answers with an action. The rules are
        deterministic, so a replay stores the bots&rsquo; actions and plays back exactly; the site checks that before it
        publishes one.
      </p>

      <h2>Tournaments</h2>
      <p>
        Every group of bots that fits a map plays it from every seating, so no bot gains from a better seat. The
        tournament runs on GitHub Actions, where the neural bots run on PyTorch as they do in training, and it runs again
        whenever a game&rsquo;s rules, maps or bots change. The site shows one replay of every pairing on every map.
      </p>

      <h2>Ratings</h2>
      <p>
        Ratings use the Elo scale: a bot rated 100 points higher is expected to win about 64% of games, 200 points about
        76%, 400 points about 91%. They are fitted to every result of the tournament at once (the Plackett&ndash;Luce
        model, which is Elo for two players and extends to three or more), so the order of matches doesn&rsquo;t matter.
        Matches on two-player and three-player maps are rated separately. The &plusmn; is a 95% interval: with more games
        it shrinks.
      </p>

      <h2>Adding a bot</h2>
      <p>
        A bot is one Python file in its game&rsquo;s folder in the repository. Open a pull request with it; once it is
        merged, the next tournament includes it.
      </p>
    </div>
  );
}
