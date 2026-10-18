// The Python that the play page runs in the browser, copied into public/play/ before
// `next dev` and `next build`. For every game in ../games:
//
//   public/play/<game>/python.json   the shared platform code and the game's code and data, as text
//   public/play/<game>/files/...     the game's other files, such as a network's weights, fetched
//                                    only when a bot needs them (the game's web_needs)
//
// A game's files are its arena_game.py and what its files.json lists, the same files a tournament
// depends on (platform/arena/contract.py). Nothing here is committed: it is rebuilt every time.

import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = path.resolve(WEB, "..");
const OUTPUT = path.join(WEB, "public", "play");

const TEXT = new Set([".py", ".json"]);

/** The game's files, relative to the repository: arena_game.py and every file files.json lists,
 * a listed folder standing for every file in it. As arena.registry.game_files reads them. */
function gameFiles(game) {
  const folder = path.join(ROOT, "games", game);
  const listed = JSON.parse(readFileSync(path.join(folder, "files.json"), "utf8"));
  const files = new Set(["arena_game.py"]);
  for (const entry of [...listed.rules, ...listed.bots]) {
    const target = path.join(folder, entry);
    if (!existsSync(target)) throw new Error(`games/${game}/files.json lists ${entry}, which does not exist`);
    if (!statSync(target).isDirectory()) files.add(entry);
    else {
      for (const found of readdirSync(target, { recursive: true, withFileTypes: true })) {
        const name = path.relative(folder, path.join(found.parentPath, found.name)).split(path.sep).join("/");
        if (found.isFile() && !name.split("/").includes("__pycache__")) files.add(name);
      }
    }
  }
  return [...files].sort().map((name) => `games/${game}/${name}`);
}

if (!existsSync(path.join(ROOT, "games")) || !existsSync(path.join(ROOT, "platform", "arena"))) {
  throw new Error(
    `The play page needs the games' Python from ${ROOT}/games and ${ROOT}/platform, which this build cannot see. ` +
      'On Vercel, keep "Include source files outside of the Root Directory in the Build Step" on (Settings > Build and Deployment > Root Directory).',
  );
}

const platform = readdirSync(path.join(ROOT, "platform", "arena"))
  .filter((name) => name.endsWith(".py"))
  .map((name) => `platform/arena/${name}`);
const games = readdirSync(path.join(ROOT, "games")).filter((name) => existsSync(path.join(ROOT, "games", name, "arena_game.py")));

rmSync(OUTPUT, { recursive: true, force: true });
for (const game of games) {
  const folder = `games/${game}`;
  const text = {};
  let binary = 0;
  for (const file of [...platform, ...gameFiles(game)]) {
    if (TEXT.has(path.extname(file))) {
      text[file] = readFileSync(path.join(ROOT, file), "utf8");
    } else {
      const target = path.join(OUTPUT, game, "files", path.posix.relative(folder, file));
      mkdirSync(path.dirname(target), { recursive: true });
      copyFileSync(path.join(ROOT, file), target);
      binary += statSync(target).size;
    }
  }
  mkdirSync(path.join(OUTPUT, game), { recursive: true });
  const bundle = JSON.stringify({ game, files: text });
  writeFileSync(path.join(OUTPUT, game, "python.json"), bundle);
  console.log(`play/${game}: ${Object.keys(text).length} Python and data files (${Math.round(bundle.length / 1024)} kB), ${Math.round(binary / 1024)} kB of other files`);
}
