// A bot's name with a small "bot" badge, so a computer player is never mistaken
// for a person once people play on the site too. Given the game and bot id, it
// also shows the bot's portrait in front of the name.

import { BotAvatar } from "./BotAvatar.tsx";

export function BotName({ name, game, id }: { name: string; game?: string; id?: string }) {
  return (
    <span className="bot-name">
      {game && id && <BotAvatar game={game} id={id} size={20} />}
      {name}
      <span className="bot-badge" title="Computer player">
        bot
      </span>
    </span>
  );
}
