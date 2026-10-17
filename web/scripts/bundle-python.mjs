// The Python that the play page runs in the browser, copied into public/play/ before
// `next dev` and `next build`. For every game in ../games:
//
//   public/play/<game>/python.json   the shared platform code and the game's code and data, as text
//   public/play/<game>/files/...     the game's other files, such as a network's weights, fetched
//                                    only when a bot needs them (the game's web_needs)
//
// Notes, tests, training scripts and pictures stay out, as they do from a tournament's inputs
// (platform/arena/publish.py). Nothing here is committed: it is rebuilt from the games every time.

import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = path.resolve(WEB, "..");
const OUTPUT = path.join(WEB, "public", "play");

const TEXT = new Set([".py", ".json"]);
const SKIPPED = new Set([".md", ".ipynb", ".txt", ".png", ".jpg", ".pyc"]);
const SKIPPED_PARTS = new Set(["tests", "scripts", "runs", "colab", "__pycache__", ".venv"]);

/** Files under a folder of the repository, relative to the repository: the ones git tracks or would
 * (so local training output it ignores stays out), else all of them. */
function listFiles(folder) {
  try {
    const listed = execFileSync("git", ["-C", ROOT, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", folder], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
    const files = listed.split("\0").filter((name) => name && existsSync(path.join(ROOT, name)));
    if (files.length) return files;
  } catch {
    // No git here (a deployment may build from a plain copy): walk the folder instead.
  }
  const found = [];
  const walk = (relative) => {
    for (const entry of readdirSync(path.join(ROOT, relative), { withFileTypes: true })) {
      const name = path.posix.join(relative, entry.name);
      if (entry.isDirectory()) {
        if (!SKIPPED_PARTS.has(entry.name)) walk(name);
      } else if (entry.isFile()) {
        found.push(name);
      }
    }
  };
  walk(folder);
  return found;
}

function wanted(file) {
  return !file.split("/").some((part) => SKIPPED_PARTS.has(part)) && !SKIPPED.has(path.extname(file));
}

if (!existsSync(path.join(ROOT, "games")) || !existsSync(path.join(ROOT, "platform", "arena"))) {
  throw new Error(
    `The play page needs the games' Python from ${ROOT}/games and ${ROOT}/platform, which this build cannot see. ` +
      'On Vercel, keep "Include source files outside of the Root Directory in the Build Step" on (Settings > Build and Deployment > Root Directory).',
  );
}

const platform = listFiles("platform/arena").filter((file) => file.endsWith(".py") && wanted(file));
const games = readdirSync(path.join(ROOT, "games")).filter((name) => existsSync(path.join(ROOT, "games", name, "arena_game.py")));

rmSync(OUTPUT, { recursive: true, force: true });
for (const game of games) {
  const folder = `games/${game}`;
  const files = listFiles(folder).filter(wanted);
  const text = {};
  let binary = 0;
  for (const file of [...platform, ...files]) {
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
