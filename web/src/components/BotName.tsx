// A bot's name with a small "bot" badge, so a computer player is never mistaken
// for a person once people play on the site too.

export function BotName({ name }: { name: string }) {
  return (
    <span className="bot-name">
      {name}
      <span className="bot-badge" title="Computer player">
        bot
      </span>
    </span>
  );
}
